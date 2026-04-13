use std::ffi::CString;

use bevy_asset::AssetLoader;
use bevy_ecs::world::FromWorld;
use pumicite::{
    Allocator, HasDevice,
    ash::{ext::debug_utils::Meta as DebugUtilsExt, vk},
    debug::DebugObject,
    image::{Image, ImageExt},
};

use super::{ImageLoadingError, TextureAsset, TextureLoadPreferences};
use crate::{DescriptorHeap, staging::AsyncTransfer};
pub use png::DecodingError as PngDecodingError;
pub struct PngLoader {
    allocator: Allocator,
    heap: Option<DescriptorHeap>,
    transfer: AsyncTransfer,
}
impl FromWorld for PngLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
            transfer: world.resource::<AsyncTransfer>().clone(),
            heap: world.get_resource().cloned(),
        }
    }
}

impl AssetLoader for PngLoader {
    type Asset = TextureAsset;
    type Settings = TextureLoadPreferences;
    type Error = ImageLoadingError;
    fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext,
    ) -> impl bevy_tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        async move {
            let mut img_data = Vec::new();
            reader.read_to_end(&mut img_data).await?;
            let decoder = png::Decoder::new(std::io::Cursor::new(img_data));
            let mut reader = decoder.read_info()?;
            let color_type = reader.output_color_type();

            let num_frames = if reader.info().is_animated() {
                reader.info().animation_control().unwrap().num_frames
            } else {
                1
            };

            let format = {
                use png::BitDepth;
                use png::ColorType::*;
                use vk::Format;
                match color_type {
                    (Grayscale, BitDepth::Eight) => Format::R8_UNORM,
                    (Rgb | Rgba, BitDepth::Four) => Format::R4G4B4A4_UNORM_PACK16,
                    (Rgb | Rgba, BitDepth::Eight) => Format::R8G8B8A8_UNORM,
                    (GrayscaleAlpha, BitDepth::Four) => Format::R4G4_UNORM_PACK8,
                    (GrayscaleAlpha, BitDepth::Eight) => Format::R8G8_UNORM,
                    _ => return Err(ImageLoadingError::UnknownTextureColorType),
                }
            };
            let mut texture = Image::new_private(
                self.allocator.clone(),
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: vk::Extent3D {
                        width: reader.info().width,
                        height: reader.info().height,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: num_frames,
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    ..Default::default()
                },
            )?;
            if self
                .allocator
                .device()
                .get_extension::<DebugUtilsExt>()
                .is_ok()
            {
                let name: String = load_context
                    .asset_path()
                    .path()
                    .as_os_str()
                    .to_string_lossy()
                    .to_string();
                if let Ok(name) = CString::new(name) {
                    texture.set_name(name.as_c_str());
                }
            }

            let bpp = reader.info().bytes_per_pixel();

            let allocation_info = texture.allocation_info();
            tracing::info!(
                "Loading png image {} ({}x{}) sized {:.2} MB{} with format {:?}, mapped to Vulkan format {:?}",
                load_context.asset_path(),
                reader.info().width,
                reader.info().height,
                allocation_info.allocation_info.size as f32 / 1024.0 / 1024.0,
                if allocation_info.dedicated_memory {
                    " dedicated"
                } else {
                    ""
                },
                color_type,
                format
            );

            let mut batch = self.transfer.batch().await?;
            let mut allocator = self.allocator.clone();
            texture
                .update_contents_async::<_, ImageLoadingError>(
                    async |slice| -> Result<(), ImageLoadingError> {
                        let num_bytes_per_frame = reader.info().width as usize
                            * reader.info().height as usize
                            * bpp as usize;

                        match color_type {
                            (png::ColorType::Rgb, png::BitDepth::Eight) => {
                                let dst_bytes_per_frame = reader.info().width as usize
                                    * reader.info().height as usize
                                    * 4;
                                let mut buffer = vec![0_u8; num_bytes_per_frame];
                                for i in 0..num_frames {
                                    let dst = &mut slice[i as usize * dst_bytes_per_frame
                                        ..(i as usize + 1) * dst_bytes_per_frame];
                                    reader.next_frame(&mut buffer)?;
                                    for (dst, src) in dst.chunks_exact_mut(4).zip(buffer.chunks(3))
                                    {
                                        dst[0..3].copy_from_slice(src);
                                        dst[3] = 255;
                                    }
                                }
                            }
                            (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
                                let dst_bytes_per_frame = reader.info().width as usize
                                    * reader.info().height as usize
                                    * 8;
                                let mut buffer = vec![0_u8; num_bytes_per_frame];
                                for i in 0..num_frames {
                                    let dst = &mut slice[i as usize * dst_bytes_per_frame
                                        ..(i as usize + 1) * dst_bytes_per_frame];
                                    reader.next_frame(&mut buffer)?;
                                    for (dst, src) in dst.chunks_exact_mut(8).zip(buffer.chunks(6))
                                    {
                                        dst[0..6].copy_from_slice(src);
                                        dst[6..8].copy_from_slice(&[0xFF, 0xFF]);
                                    }
                                }
                            }
                            _ => {
                                for i in 0..num_frames {
                                    let dst = &mut slice[i as usize * num_bytes_per_frame
                                        ..(i as usize + 1) * num_bytes_per_frame];
                                    reader
                                        .next_frame(dst)
                                        .map_err(ImageLoadingError::PngDecodingError)?;
                                }
                            }
                        };
                        Ok::<_, ImageLoadingError>(())
                    },
                    &mut batch,
                    &mut allocator,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
                .await?;
            batch.submit().await?;
            Ok(TextureAsset::new(
                texture,
                if settings.register_bindless {
                    self.heap
                        .as_ref()
                        .map(DescriptorHeap::resource_heap)
                        .cloned()
                } else {
                    None
                },
            )?)
        }
    }

    fn extensions(&self) -> &[&str] {
        &["png"]
    }
}
