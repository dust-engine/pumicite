use std::ffi::CString;

use pumicite::image::ImageExt;

use super::*;

#[derive(thiserror::Error, Debug)]
pub enum DdsError {
    #[error("Failed to parse the ktx texture file: {0}")]
    DdsParseError(#[from] ddsfile::Error),

    #[error("VulkanError")]
    VulkanError(#[from] vk::Result),

    #[error("IoError")]
    IoError(#[from] std::io::Error),

    #[error("FormatRequiresTranscodingError")]
    FormatRequiresTranscodingError,
}

pub struct DdsLoader {
    allocator: Allocator,
    transfer: AsyncTransfer,
    heap: Option<DescriptorHeap>,
}
impl FromWorld for DdsLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            allocator: world.resource::<Allocator>().clone(),
            transfer: world.resource::<AsyncTransfer>().clone(),
            heap: world.get_resource().cloned(),
        }
    }
}

impl AssetLoader for DdsLoader {
    type Asset = TextureAsset;

    type Settings = TextureLoadPreferences;

    type Error = DdsError;

    fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext,
    ) -> impl bevy_tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        async {
            use byteorder::{LittleEndian, ReadBytesExt};
            let mut buf = [0_u8; 128];
            reader.read_exact(&mut buf).await?;
            let mut rc = std::io::Cursor::new(&buf);

            let magic = rc.read_u32::<LittleEndian>()?;
            if magic != 0x20534444 {
                return Err(DdsError::DdsParseError(ddsfile::Error::BadMagicNumber));
            }
            let header = ddsfile::Header::read(rc)?;

            let mut header10_buf = [0_u8; 5 * 4];
            reader.read_exact(&mut header10_buf).await?;
            let mut rc = std::io::Cursor::new(&header10_buf);
            let header10 = if header.spf.fourcc == Some(ddsfile::FourCC(<ddsfile::FourCC>::DX10)) {
                Some(ddsfile::Header10::read(&mut rc)?)
            } else {
                None
            };

            let dds = ddsfile::Dds {
                header,
                header10,
                data: Vec::new(),
            };

            let mut image = Image::new_upload(
                self.allocator.clone(),
                &vk::ImageCreateInfo {
                    image_type: if dds.get_depth() > 1 {
                        vk::ImageType::TYPE_3D
                    } else if dds.get_height() > 1 {
                        vk::ImageType::TYPE_2D
                    } else {
                        vk::ImageType::TYPE_1D
                    },
                    format: dds_format_to_texture_format(&dds, settings.is_srgb)?,
                    extent: vk::Extent3D {
                        width: dds.get_width(),
                        height: dds.get_height(),
                        depth: dds.get_depth(),
                    },
                    mip_levels: dds.get_num_mipmap_levels(),
                    array_layers: dds.get_num_array_layers(),
                    samples: vk::SampleCountFlags::TYPE_1,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage: vk::ImageUsageFlags::SAMPLED,
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
                    image.set_name(name.as_c_str());
                }
            }

            let mut allocator = self.allocator.clone();
            let mut batch = self.transfer.batch().await?;
            image
                .update_contents_async(
                    async |slice| {
                        if dds.header10.is_some() {
                            reader.read_exact(slice).await?;
                        } else {
                            slice[0..header10_buf.len()].copy_from_slice(&header10_buf);
                            reader.read_exact(&mut slice[header10_buf.len()..]).await?;
                        }
                        Ok::<(), DdsError>(())
                    },
                    &mut batch,
                    &mut allocator,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
                .await?;
            batch.submit().await?;

            tracing::info!("Loading {}", load_context.asset_path());

            Ok(TextureAsset::new(
                image,
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
        &["dds"]
    }
}

pub fn dds_format_to_texture_format(
    dds: &ddsfile::Dds,
    is_srgb: bool,
) -> Result<vk::Format, DdsError> {
    use ddsfile::DxgiFormat;
    Ok(if let Some(dxgi_format) = dds.get_dxgi_format() {
        match dxgi_format {
            DxgiFormat::R32G32B32A32_Typeless | DxgiFormat::R32G32B32A32_Float => {
                vk::Format::R32G32B32A32_SFLOAT
            }
            DxgiFormat::R32G32B32A32_UInt => vk::Format::R32G32B32A32_UINT,
            DxgiFormat::R32G32B32A32_SInt => vk::Format::R32G32B32A32_SINT,
            DxgiFormat::R16G16B16A16_Typeless | DxgiFormat::R16G16B16A16_Float => {
                vk::Format::R16G16B16A16_SFLOAT
            }
            DxgiFormat::R16G16B16A16_UNorm => vk::Format::R16G16B16A16_UNORM,
            DxgiFormat::R16G16B16A16_UInt => vk::Format::R16G16B16A16_UINT,
            DxgiFormat::R16G16B16A16_SNorm => vk::Format::R16G16B16A16_SNORM,
            DxgiFormat::R16G16B16A16_SInt => vk::Format::R16G16B16A16_SINT,
            DxgiFormat::R32G32_Typeless | DxgiFormat::R32G32_Float => vk::Format::R32G32_SFLOAT,
            DxgiFormat::R32G32_UInt => vk::Format::R32G32_UINT,
            DxgiFormat::R32G32_SInt => vk::Format::R32G32_SINT,
            DxgiFormat::R10G10B10A2_Typeless | DxgiFormat::R10G10B10A2_UNorm => {
                vk::Format::A2B10G10R10_UNORM_PACK32
            }
            DxgiFormat::R11G11B10_Float => vk::Format::B10G11R11_UFLOAT_PACK32,
            DxgiFormat::R8G8B8A8_Typeless
            | DxgiFormat::R8G8B8A8_UNorm
            | DxgiFormat::R8G8B8A8_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                }
            }
            DxgiFormat::R8G8B8A8_UInt => vk::Format::R8G8B8A8_UINT,
            DxgiFormat::R8G8B8A8_SNorm => vk::Format::R8G8B8A8_SNORM,
            DxgiFormat::R8G8B8A8_SInt => vk::Format::R8G8B8A8_SINT,
            DxgiFormat::R16G16_Typeless | DxgiFormat::R16G16_Float => vk::Format::R16G16_SFLOAT,
            DxgiFormat::R16G16_UNorm => vk::Format::R16G16_UNORM,
            DxgiFormat::R16G16_UInt => vk::Format::R16G16_UINT,
            DxgiFormat::R16G16_SNorm => vk::Format::R16G16_SNORM,
            DxgiFormat::R16G16_SInt => vk::Format::R16G16_SINT,
            DxgiFormat::R32_Typeless | DxgiFormat::R32_Float => vk::Format::R32_SFLOAT,
            DxgiFormat::D32_Float => vk::Format::D32_SFLOAT,
            DxgiFormat::R32_UInt => vk::Format::R32_UINT,
            DxgiFormat::R32_SInt => vk::Format::R32_SINT,
            DxgiFormat::R24G8_Typeless | DxgiFormat::D24_UNorm_S8_UInt => {
                vk::Format::D24_UNORM_S8_UINT
            }
            DxgiFormat::R24_UNorm_X8_Typeless => vk::Format::X8_D24_UNORM_PACK32,
            DxgiFormat::R8G8_Typeless | DxgiFormat::R8G8_UNorm => vk::Format::R8G8_UNORM,
            DxgiFormat::R8G8_UInt => vk::Format::R8G8_UINT,
            DxgiFormat::R8G8_SNorm => vk::Format::R8G8_SNORM,
            DxgiFormat::R8G8_SInt => vk::Format::R8G8_SINT,
            DxgiFormat::R16_Typeless | DxgiFormat::R16_Float => vk::Format::R16_SFLOAT,
            DxgiFormat::R16_UNorm => vk::Format::R16_UNORM,
            DxgiFormat::R16_UInt => vk::Format::R16_UINT,
            DxgiFormat::R16_SNorm => vk::Format::R16_SNORM,
            DxgiFormat::R16_SInt => vk::Format::R16_SINT,
            DxgiFormat::R8_Typeless | DxgiFormat::R8_UNorm => vk::Format::R8_UNORM,
            DxgiFormat::R8_UInt => vk::Format::R8_UINT,
            DxgiFormat::R8_SNorm => vk::Format::R8_SNORM,
            DxgiFormat::R8_SInt => vk::Format::R8_SINT,
            DxgiFormat::R9G9B9E5_SharedExp => vk::Format::E5B9G9R9_UFLOAT_PACK32,
            DxgiFormat::BC1_Typeless | DxgiFormat::BC1_UNorm | DxgiFormat::BC1_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::BC1_RGB_SRGB_BLOCK
                } else {
                    vk::Format::BC1_RGB_UNORM_BLOCK
                }
            }
            DxgiFormat::BC2_Typeless | DxgiFormat::BC2_UNorm | DxgiFormat::BC2_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::BC2_UNORM_BLOCK
                } else {
                    vk::Format::BC2_SRGB_BLOCK
                }
            }
            DxgiFormat::BC3_Typeless | DxgiFormat::BC3_UNorm | DxgiFormat::BC3_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::BC3_SRGB_BLOCK
                } else {
                    vk::Format::BC3_UNORM_BLOCK
                }
            }
            DxgiFormat::BC4_Typeless | DxgiFormat::BC4_UNorm => vk::Format::BC4_UNORM_BLOCK,
            DxgiFormat::BC4_SNorm => vk::Format::BC4_SNORM_BLOCK,
            DxgiFormat::BC5_Typeless | DxgiFormat::BC5_UNorm => vk::Format::BC5_UNORM_BLOCK,
            DxgiFormat::BC5_SNorm => vk::Format::BC5_SNORM_BLOCK,
            DxgiFormat::B8G8R8A8_UNorm
            | DxgiFormat::B8G8R8A8_Typeless
            | DxgiFormat::B8G8R8A8_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::B8G8R8A8_SRGB
                } else {
                    vk::Format::B8G8R8A8_UNORM
                }
            }

            DxgiFormat::BC6H_Typeless | DxgiFormat::BC6H_UF16 => vk::Format::BC6H_UFLOAT_BLOCK,
            DxgiFormat::BC6H_SF16 => vk::Format::BC6H_SFLOAT_BLOCK,
            DxgiFormat::BC7_Typeless | DxgiFormat::BC7_UNorm | DxgiFormat::BC7_UNorm_sRGB => {
                if is_srgb {
                    vk::Format::BC7_SRGB_BLOCK
                } else {
                    vk::Format::BC7_UNORM_BLOCK
                }
            }
            _ => return Err(DdsError::FormatRequiresTranscodingError),
        }
    } else {
        return Err(DdsError::FormatRequiresTranscodingError);
    })
}
