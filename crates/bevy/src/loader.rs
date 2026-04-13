//! Image loaders
//! This module largely replaces bevy_image and offers a direct way to load images.

#[cfg(feature = "dds")]
mod dds;
#[cfg(feature = "png")]
mod png;

pub use dds::*;
pub use png::*;

use std::ops::Deref;

use bevy_asset::{Asset, AssetLoader, AsyncReadExt, io::AsyncSeekForwardExt};
use bevy_ecs::world::FromWorld;
use bevy_reflect::TypePath;
use pumicite::{ash::VkResult, bindless::ResourceHeap, image::FullImageView, prelude::*};
use serde::{Deserialize, Serialize};

use crate::{DescriptorHeap, staging::AsyncTransfer};
use ash::ext::debug_utils::Meta as DebugUtilsExt;

/// A readonly texture asset loaded once and never written to.
#[derive(Asset, TypePath)]
pub struct TextureAsset {
    heap: Option<ResourceHeap>,
    image: FullImageView<Image>,
    handle: u32,
}
impl Deref for TextureAsset {
    type Target = FullImageView<Image>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}
impl TextureAsset {
    pub fn new(image: Image, heap: Option<ResourceHeap>) -> VkResult<Self> {
        let image = image.create_full_view()?;
        let handle = if let Some(heap) = heap.as_ref() {
            heap.add_image(image.image(), pumicite::bindless::ImageAccessMode::Sampled)?
        } else {
            u32::MAX
        };
        Ok(Self {
            image,
            heap,
            handle,
        })
    }
    pub fn handle(&self) -> u32 {
        self.handle
    }
}
impl Drop for TextureAsset {
    fn drop(&mut self) {
        if let Some(heap) = &self.heap {
            heap.remove(self.handle);
        }
    }
}

#[cfg(feature = "image")]
pub use img_loader::*;

#[cfg(feature = "image")]
mod img_loader {
    use std::ffi::CString;

    use super::*;

    #[derive(thiserror::Error, Debug)]
    pub enum ImageLoadingError {
        #[error("Unknown color type")]
        UnknownTextureColorType,

        #[error("Image error: {0:?}")]
        #[cfg(feature = "image")]
        ImageError(#[from] image::ImageError),

        #[error("Png decoding error: {0:?}")]
        #[cfg(feature = "png")]
        PngDecodingError(#[from] PngDecodingError),

        #[error("VulkanError")]
        VulkanError(#[from] vk::Result),

        #[error("IoError")]
        IoError(#[from] std::io::Error),

        #[error("FormatRequiresTranscodingError")]
        FormatRequiresTranscodingError,
    }

    pub struct ImageLoader {
        allocator: Allocator,
        heap: Option<DescriptorHeap>,
        transfer: AsyncTransfer,
    }
    impl FromWorld for ImageLoader {
        fn from_world(world: &mut bevy_ecs::world::World) -> Self {
            Self {
                allocator: world.resource::<Allocator>().clone(),
                transfer: world.resource::<AsyncTransfer>().clone(),
                heap: world.get_resource().cloned(),
            }
        }
    }

    impl AssetLoader for ImageLoader {
        type Asset = TextureAsset;

        type Settings = TextureLoadPreferences;

        type Error = ImageLoadingError;
        fn load(
            &self,
            reader: &mut dyn bevy_asset::io::Reader,
            settings: &Self::Settings,
            load_context: &mut bevy_asset::LoadContext,
        ) -> impl bevy_tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>>
        {
            use image::ColorType;
            async {
                let mut allocator = self.allocator.clone();
                let mut data = Vec::new();
                reader.read_to_end(&mut data).await?;
                let mut image_reader = image::ImageReader::new(std::io::Cursor::new(&data));
                match load_context
                    .path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("")
                {
                    "png" => image_reader.set_format(image::ImageFormat::Png),
                    "jpg" | "jpeg" => image_reader.set_format(image::ImageFormat::Jpeg),
                    "tif" | "tiff" => image_reader.set_format(image::ImageFormat::Tiff),
                    _ => image_reader = image_reader.with_guessed_format()?,
                }
                let image = image_reader.decode()?;

                let format = match (image.color(), settings.is_srgb) {
                    (ColorType::L8, false) => vk::Format::R8_UNORM,
                    (ColorType::La8, false) => vk::Format::R8G8_UNORM,
                    (ColorType::Rgb8, false) => vk::Format::R8G8B8A8_UNORM,
                    (ColorType::Rgba8, false) => vk::Format::R8G8B8A8_UNORM,
                    (ColorType::L8, true) => vk::Format::R8_SRGB,
                    (ColorType::La8, true) => vk::Format::R8G8_SRGB,
                    (ColorType::Rgb8, true) => vk::Format::R8G8B8A8_SRGB,
                    (ColorType::Rgba8, true) => vk::Format::R8G8B8A8_SRGB,
                    (ColorType::L16, _) => vk::Format::R16_UNORM,
                    (ColorType::La16, _) => vk::Format::R16G16_UNORM,
                    (ColorType::Rgb16, _) => vk::Format::R16G16B16A16_UNORM,
                    (ColorType::Rgba16, _) => vk::Format::R16G16B16A16_UNORM,
                    (ColorType::Rgb32F, _) => vk::Format::R32G32B32A32_SFLOAT,
                    (ColorType::Rgba32F, _) => vk::Format::R32G32B32A32_SFLOAT,
                    _ => return Err(ImageLoadingError::UnknownTextureColorType),
                };
                let mut texture = Image::new_private(
                    self.allocator.clone(),
                    &vk::ImageCreateInfo {
                        image_type: vk::ImageType::TYPE_2D,
                        format,
                        extent: vk::Extent3D {
                            width: image.width(),
                            height: image.height(),
                            depth: 1,
                        },
                        mip_levels: 1,
                        array_layers: 1,
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
                let allocation_info = texture.allocation_info();
                tracing::info!(
                    "Loading image ({}x{}) sized {:.2} MB{} with format {:?}, mapped to Vulkan format {:?}",
                    image.width(),
                    image.height(),
                    allocation_info.allocation_info.size as f32 / 1024.0 / 1024.0,
                    if allocation_info.dedicated_memory {
                        " dedicated"
                    } else {
                        ""
                    },
                    image.color(),
                    format
                );
                let mut batch = self.transfer.batch().await?;
                texture
                    .update_contents_async(
                        async |slice| {
                            match image.color() {
                                ColorType::Rgb8 => {
                                    for (dst, src) in
                                        slice.chunks_exact_mut(4).zip(image.as_bytes().chunks(3))
                                    {
                                        dst[0..3].copy_from_slice(src);
                                    }
                                }
                                ColorType::Rgb16 => {
                                    for (dst, src) in
                                        slice.chunks_exact_mut(8).zip(image.as_bytes().chunks(6))
                                    {
                                        dst[0..6].copy_from_slice(src);
                                    }
                                }
                                ColorType::Rgb32F => {
                                    for (dst, src) in
                                        slice.chunks_exact_mut(16).zip(image.as_bytes().chunks(12))
                                    {
                                        dst[0..12].copy_from_slice(src);
                                    }
                                }
                                _ => {
                                    slice.copy_from_slice(image.as_bytes());
                                }
                            }
                            Ok::<_, vk::Result>(())
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
            &["tif", "jpg", "exr"]
        }
    }
}

#[cfg(feature = "ktx2")]
#[derive(thiserror::Error, Debug)]
pub enum KtxError {
    #[error("Failed to parse the ktx texture file: {0}")]
    KtxParseError(#[from] ktx2::ParseError),
    #[error("HeaderReadIOError")]
    ReadIOError,

    #[error("VulkanError")]
    VulkanError(#[from] vk::Result),
}
#[cfg(feature = "ktx2")]
pub struct KtxLoader {
    allocator: Allocator,
}
#[cfg(feature = "ktx2")]
impl FromWorld for KtxLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
        }
    }
}

#[cfg(feature = "ktx2")]
mod settings_serde {
    use pumicite::ash::vk;
    use serde::{Deserialize, Serialize};
    macro_rules! define_serde {
        ($type_name: ty, $serde_name: ident, $deser_name: ident, $primitive_type: ty) => {
            pub fn $serde_name<S: serde::Serializer>(
                id: &$type_name,
                s: S,
            ) -> Result<S::Ok, S::Error> {
                id.as_raw().serialize(s)
            }
            pub fn $deser_name<'de, D>(d: D) -> Result<$type_name, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                Ok(<$type_name>::from_raw(<$primitive_type>::deserialize(d)?))
            }
        };
    }
    define_serde!(vk::ImageUsageFlags, serde_usage, deser_usage, u32);
    define_serde!(
        vk::ImageCreateFlags,
        serde_create_flags,
        deser_create_flags,
        u32
    );
    define_serde!(vk::ImageTiling, serde_tiling, deser_tiling, i32);
    define_serde!(vk::ImageLayout, serde_layout, deser_layout, i32);
}

#[cfg(feature = "ktx2")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct KtxLoaderSettings {
    #[serde(
        serialize_with = "settings_serde::serde_usage",
        deserialize_with = "settings_serde::deser_usage"
    )]
    pub usage: vk::ImageUsageFlags,
    #[serde(
        serialize_with = "settings_serde::serde_create_flags",
        deserialize_with = "settings_serde::deser_create_flags"
    )]
    pub flags: vk::ImageCreateFlags,
    #[serde(
        serialize_with = "settings_serde::serde_tiling",
        deserialize_with = "settings_serde::deser_tiling"
    )]
    pub tiling: vk::ImageTiling,
    #[serde(
        serialize_with = "settings_serde::serde_layout",
        deserialize_with = "settings_serde::deser_layout"
    )]
    pub layout: vk::ImageLayout,
}
#[cfg(feature = "ktx2")]
impl Default for KtxLoaderSettings {
    fn default() -> Self {
        Self {
            usage: vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
            tiling: vk::ImageTiling::OPTIMAL,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}
#[cfg(feature = "ktx2")]
impl AssetLoader for KtxLoader {
    type Asset = TextureAsset;

    type Settings = KtxLoaderSettings;

    type Error = KtxError;

    fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        settings: &Self::Settings,
        _load_context: &mut bevy_asset::LoadContext,
    ) -> impl bevy_tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        async {
            let mut buffer = [0_u8; ktx2::Header::LENGTH];
            reader
                .read_exact(&mut buffer)
                .await
                .map_err(|_| KtxError::ReadIOError)?;

            let mut header = ktx2::Header::from_bytes(buffer.as_slice().try_into().unwrap())?;
            let format = header
                .format
                .map(|x| vk::Format::from_raw(x.value() as i32))
                .unwrap_or(vk::Format::UNDEFINED);
            let image_type = if header.pixel_height == 0 && header.pixel_depth == 0 {
                vk::ImageType::TYPE_1D
            } else if header.pixel_depth > 0 {
                vk::ImageType::TYPE_3D
            } else {
                vk::ImageType::TYPE_2D
            };
            header.level_count = header.level_count.max(1);
            header.layer_count = header.layer_count.max(1);
            if header.layer_count != 1 && header.layer_count != 6 {
                return Err(KtxError::KtxParseError(ktx2::ParseError::ZeroFaceCount));
            }
            let mut image_create_info = vk::ImageCreateInfo {
                format,
                extent: vk::Extent3D {
                    width: header.pixel_width,
                    height: header.pixel_height,
                    depth: header.pixel_depth,
                },
                mip_levels: header.level_count,
                array_layers: header.layer_count,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: settings.tiling,
                usage: settings.usage,
                image_type,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            if header.face_count == 6 {
                image_create_info.array_layers = header.layer_count * 6;
            }
            let device_image = Image::new_private(self.allocator.clone(), &image_create_info)?;
            let _device_image = GPUMutex::new(device_image);

            let level_indexes_size = ktx2::LevelIndex::LENGTH * header.level_count as usize;
            let mut level_indexes = vec![0_u8; level_indexes_size];
            reader
                .read_exact(&mut level_indexes)
                .await
                .map_err(|_| KtxError::ReadIOError)?;
            debug_assert!(
                header.index.dfd_byte_offset as usize
                    == ktx2::LevelIndex::LENGTH * header.level_count as usize
                        + ktx2::Header::LENGTH
            );
            let _level_indexes = unsafe {
                std::slice::from_raw_parts(
                    level_indexes.as_ptr() as *const ktx2::LevelIndex,
                    header.level_count as usize,
                )
            };

            reader
                .seek_forward(
                    header.index.sgd_byte_offset
                        - ktx2::Header::LENGTH as u64
                        - level_indexes_size as u64
                        + header.index.sgd_byte_length,
                )
                .await
                .map_err(|_| KtxError::ReadIOError)?; // Skip everything to "mip level array"

            let format_info = pumicite::types::format::Format::from(format).properties();
            let total_data_size = (0..header.level_count)
                .map(|mip_level| {
                    let width = (header.pixel_width >> mip_level).max(1);
                    let height = (header.pixel_height >> mip_level).max(1);
                    let depth = (header.pixel_depth >> mip_level).max(1);
                    let texel_count = width * height * depth;
                    let byte_count = format_info.block_size * texel_count;
                    byte_count as u64 * header.layer_count as u64 * header.face_count as u64
                })
                .sum();

            let _staging_buffer = Buffer::new_host(
                self.allocator.clone(),
                total_data_size,
                format_info.block_size as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
            )?;

            panic!()
        }
    }
    fn extensions(&self) -> &[&str] {
        &["ktx"]
    }
}

#[derive(Serialize, Deserialize)]
pub struct TextureLoadPreferences {
    is_srgb: bool,
    register_bindless: bool,
}
impl Default for TextureLoadPreferences {
    fn default() -> Self {
        Self {
            is_srgb: false,
            register_bindless: true,
        }
    }
}
