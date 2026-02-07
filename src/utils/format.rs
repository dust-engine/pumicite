use ash::vk;
use glam::{Mat3, UVec2, UVec3, Vec2};
use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "bevy", derive(bevy_reflect::Reflect))]
pub enum Format {
    #[default]
    UNDEFINED = 0,
    R4G4_UNORM_PACK8 = 1,
    R4G4B4A4_UNORM_PACK16 = 2,
    B4G4R4A4_UNORM_PACK16 = 3,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R5G5B5A1_UNORM_PACK16 = 6,
    B5G5R5A1_UNORM_PACK16 = 7,
    A1R5G5B5_UNORM_PACK16 = 8,
    R8_UNORM = 9,
    R8_SNORM = 10,
    R8_USCALED = 11,
    R8_SSCALED = 12,
    R8_UINT = 13,
    R8_SINT = 14,
    R8_SRGB = 15,
    R8G8_UNORM = 16,
    R8G8_SNORM = 17,
    R8G8_USCALED = 18,
    R8G8_SSCALED = 19,
    R8G8_UINT = 20,
    R8G8_SINT = 21,
    R8G8_SRGB = 22,
    R8G8B8_UNORM = 23,
    R8G8B8_SNORM = 24,
    R8G8B8_USCALED = 25,
    R8G8B8_SSCALED = 26,
    R8G8B8_UINT = 27,
    R8G8B8_SINT = 28,
    R8G8B8_SRGB = 29,
    B8G8R8_UNORM = 30,
    B8G8R8_SNORM = 31,
    B8G8R8_USCALED = 32,
    B8G8R8_SSCALED = 33,
    B8G8R8_UINT = 34,
    B8G8R8_SINT = 35,
    B8G8R8_SRGB = 36,
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SNORM = 38,
    R8G8B8A8_USCALED = 39,
    R8G8B8A8_SSCALED = 40,
    R8G8B8A8_UINT = 41,
    R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SNORM = 45,
    B8G8R8A8_USCALED = 46,
    B8G8R8A8_SSCALED = 47,
    B8G8R8A8_UINT = 48,
    B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB = 50,
    A8B8G8R8_UNORM_PACK32 = 51,
    A8B8G8R8_SNORM_PACK32 = 52,
    A8B8G8R8_USCALED_PACK32 = 53,
    A8B8G8R8_SSCALED_PACK32 = 54,
    A8B8G8R8_UINT_PACK32 = 55,
    A8B8G8R8_SINT_PACK32 = 56,
    A8B8G8R8_SRGB_PACK32 = 57,
    A2R10G10B10_UNORM_PACK32 = 58,
    A2R10G10B10_SNORM_PACK32 = 59,
    A2R10G10B10_USCALED_PACK32 = 60,
    A2R10G10B10_SSCALED_PACK32 = 61,
    A2R10G10B10_UINT_PACK32 = 62,
    A2R10G10B10_SINT_PACK32 = 63,
    A2B10G10R10_UNORM_PACK32 = 64,
    A2B10G10R10_SNORM_PACK32 = 65,
    A2B10G10R10_USCALED_PACK32 = 66,
    A2B10G10R10_SSCALED_PACK32 = 67,
    A2B10G10R10_UINT_PACK32 = 68,
    A2B10G10R10_SINT_PACK32 = 69,
    R16_UNORM = 70,
    R16_SNORM = 71,
    R16_USCALED = 72,
    R16_SSCALED = 73,
    R16_UINT = 74,
    R16_SINT = 75,
    R16_SFLOAT = 76,
    R16G16_UNORM = 77,
    R16G16_SNORM = 78,
    R16G16_USCALED = 79,
    R16G16_SSCALED = 80,
    R16G16_UINT = 81,
    R16G16_SINT = 82,
    R16G16_SFLOAT = 83,
    R16G16B16_UNORM = 84,
    R16G16B16_SNORM = 85,
    R16G16B16_USCALED = 86,
    R16G16B16_SSCALED = 87,
    R16G16B16_UINT = 88,
    R16G16B16_SINT = 89,
    R16G16B16_SFLOAT = 90,
    R16G16B16A16_UNORM = 91,
    R16G16B16A16_SNORM = 92,
    R16G16B16A16_USCALED = 93,
    R16G16B16A16_SSCALED = 94,
    R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96,
    R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98,
    R32_SINT = 99,
    R32_SFLOAT = 100,
    R32G32_UINT = 101,
    R32G32_SINT = 102,
    R32G32_SFLOAT = 103,
    R32G32B32_UINT = 104,
    R32G32B32_SINT = 105,
    R32G32B32_SFLOAT = 106,
    R32G32B32A32_UINT = 107,
    R32G32B32A32_SINT = 108,
    R32G32B32A32_SFLOAT = 109,
    R64_UINT = 110,
    R64_SINT = 111,
    R64_SFLOAT = 112,
    R64G64_UINT = 113,
    R64G64_SINT = 114,
    R64G64_SFLOAT = 115,
    R64G64B64_UINT = 116,
    R64G64B64_SINT = 117,
    R64G64B64_SFLOAT = 118,
    R64G64B64A64_UINT = 119,
    R64G64B64A64_SINT = 120,
    R64G64B64A64_SFLOAT = 121,
    B10G11R11_UFLOAT_PACK32 = 122,
    E5B9G9R9_UFLOAT_PACK32 = 123,
    D16_UNORM = 124,
    X8_D24_UNORM_PACK32 = 125,
    D32_SFLOAT = 126,
    S8_UINT = 127,
    D16_UNORM_S8_UINT = 128,
    D24_UNORM_S8_UINT = 129,
    D32_SFLOAT_S8_UINT = 130,
    BC1_RGB_UNORM_BLOCK = 131,
    BC1_RGB_SRGB_BLOCK = 132,
    BC1_RGBA_UNORM_BLOCK = 133,
    BC1_RGBA_SRGB_BLOCK = 134,
    BC2_UNORM_BLOCK = 135,
    BC2_SRGB_BLOCK = 136,
    BC3_UNORM_BLOCK = 137,
    BC3_SRGB_BLOCK = 138,
    BC4_UNORM_BLOCK = 139,
    BC4_SNORM_BLOCK = 140,
    BC5_UNORM_BLOCK = 141,
    BC5_SNORM_BLOCK = 142,
    BC6H_UFLOAT_BLOCK = 143,
    BC6H_SFLOAT_BLOCK = 144,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
    ETC2_R8G8B8_UNORM_BLOCK = 147,
    ETC2_R8G8B8_SRGB_BLOCK = 148,
    ETC2_R8G8B8A1_UNORM_BLOCK = 149,
    ETC2_R8G8B8A1_SRGB_BLOCK = 150,
    ETC2_R8G8B8A8_UNORM_BLOCK = 151,
    ETC2_R8G8B8A8_SRGB_BLOCK = 152,
    EAC_R11_UNORM_BLOCK = 153,
    EAC_R11_SNORM_BLOCK = 154,
    EAC_R11G11_UNORM_BLOCK = 155,
    EAC_R11G11_SNORM_BLOCK = 156,
    ASTC_4x4_UNORM_BLOCK = 157,
    ASTC_4x4_SRGB_BLOCK = 158,
    ASTC_5x4_UNORM_BLOCK = 159,
    ASTC_5x4_SRGB_BLOCK = 160,
    ASTC_5x5_UNORM_BLOCK = 161,
    ASTC_5x5_SRGB_BLOCK = 162,
    ASTC_6x5_UNORM_BLOCK = 163,
    ASTC_6x5_SRGB_BLOCK = 164,
    ASTC_6x6_UNORM_BLOCK = 165,
    ASTC_6x6_SRGB_BLOCK = 166,
    ASTC_8x5_UNORM_BLOCK = 167,
    ASTC_8x5_SRGB_BLOCK = 168,
    ASTC_8x6_UNORM_BLOCK = 169,
    ASTC_8x6_SRGB_BLOCK = 170,
    ASTC_8x8_UNORM_BLOCK = 171,
    ASTC_8x8_SRGB_BLOCK = 172,
    ASTC_10x5_UNORM_BLOCK = 173,
    ASTC_10x5_SRGB_BLOCK = 174,
    ASTC_10x6_UNORM_BLOCK = 175,
    ASTC_10x6_SRGB_BLOCK = 176,
    ASTC_10x8_UNORM_BLOCK = 177,
    ASTC_10x8_SRGB_BLOCK = 178,
    ASTC_10x10_UNORM_BLOCK = 179,
    ASTC_10x10_SRGB_BLOCK = 180,
    ASTC_12x10_UNORM_BLOCK = 181,
    ASTC_12x10_SRGB_BLOCK = 182,
    ASTC_12x12_UNORM_BLOCK = 183,
    ASTC_12x12_SRGB_BLOCK = 184,
    G8B8G8R8_422_UNORM = 1000156000,
    B8G8R8G8_422_UNORM = 1000156001,
    G8_B8_R8_3PLANE_420_UNORM = 1000156002,
    G8_B8R8_2PLANE_420_UNORM = 1000156003,
    G8_B8_R8_3PLANE_422_UNORM = 1000156004,
    G8_B8R8_2PLANE_422_UNORM = 1000156005,
    G8_B8_R8_3PLANE_444_UNORM = 1000156006,
    R10X6_UNORM_PACK16 = 1000156007,
    R10X6G10X6_UNORM_2PACK16 = 1000156008,
    R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
    G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
    B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
    G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
    G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
    G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
    G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
    G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
    R12X4_UNORM_PACK16 = 1000156017,
    R12X4G12X4_UNORM_2PACK16 = 1000156018,
    R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
    G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
    B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
    G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
    G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
    G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
    G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
    G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
    G16B16G16R16_422_UNORM = 1000156027,
    B16G16R16G16_422_UNORM = 1000156028,
    G16_B16_R16_3PLANE_420_UNORM = 1000156029,
    G16_B16R16_2PLANE_420_UNORM = 1000156030,
    G16_B16_R16_3PLANE_422_UNORM = 1000156031,
    G16_B16R16_2PLANE_422_UNORM = 1000156032,
    G16_B16_R16_3PLANE_444_UNORM = 1000156033,
    G8_B8R8_2PLANE_444_UNORM = 1000330000,
    G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16 = 1000330001,
    G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16 = 1000330002,
    G16_B16R16_2PLANE_444_UNORM = 1000330003,
    A4R4G4B4_UNORM_PACK16 = 1000340000,
    A4B4G4R4_UNORM_PACK16 = 1000340001,
    ASTC_4x4_SFLOAT_BLOCK = 1000066000,
    ASTC_5x4_SFLOAT_BLOCK = 1000066001,
    ASTC_5x5_SFLOAT_BLOCK = 1000066002,
    ASTC_6x5_SFLOAT_BLOCK = 1000066003,
    ASTC_6x6_SFLOAT_BLOCK = 1000066004,
    ASTC_8x5_SFLOAT_BLOCK = 1000066005,
    ASTC_8x6_SFLOAT_BLOCK = 1000066006,
    ASTC_8x8_SFLOAT_BLOCK = 1000066007,
    ASTC_10x5_SFLOAT_BLOCK = 1000066008,
    ASTC_10x6_SFLOAT_BLOCK = 1000066009,
    ASTC_10x8_SFLOAT_BLOCK = 1000066010,
    ASTC_10x10_SFLOAT_BLOCK = 1000066011,
    ASTC_12x10_SFLOAT_BLOCK = 1000066012,
    ASTC_12x12_SFLOAT_BLOCK = 1000066013,
    A1B5G5R5_UNORM_PACK16 = 1000470000,
    A8_UNORM = 1000470001,
    PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
    PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
    PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
    PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
    PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
    PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
    PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
    PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
    R8_BOOL_ARM = 1000460000,
    R16G16_SFIXED5_NV = 1000464000,
    R10X6_UINT_PACK16_ARM = 1000609000,
    R10X6G10X6_UINT_2PACK16_ARM = 1000609001,
    R10X6G10X6B10X6A10X6_UINT_4PACK16_ARM = 1000609002,
    R12X4_UINT_PACK16_ARM = 1000609003,
    R12X4G12X4_UINT_2PACK16_ARM = 1000609004,
    R12X4G12X4B12X4A12X4_UINT_4PACK16_ARM = 1000609005,
    R14X2_UINT_PACK16_ARM = 1000609006,
    R14X2G14X2_UINT_2PACK16_ARM = 1000609007,
    R14X2G14X2B14X2A14X2_UINT_4PACK16_ARM = 1000609008,
    R14X2_UNORM_PACK16_ARM = 1000609009,
    R14X2G14X2_UNORM_2PACK16_ARM = 1000609010,
    R14X2G14X2B14X2A14X2_UNORM_4PACK16_ARM = 1000609011,
    G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16_ARM = 1000609012,
    G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16_ARM = 1000609013,
}

impl From<Format> for vk::Format {
    fn from(value: Format) -> Self {
        vk::Format::from_raw(value as i32)
    }
}
impl From<vk::Format> for Format {
    #[rustfmt::skip]
    fn from(value: vk::Format) -> Self {
        unsafe {
            std::mem::transmute(value)
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FormatType {
    /// Value will be converted to a float in the range of [0, 1]
    UNorm,
    /// Value will be converted to as a float in the range of [-1, 1]
    SNorm,
    /// Value will be intepreted as an unsigned integer, then cast to a float with the same magnitude.
    /// For example, R8_USCALED will be converted to a float in the range of [0, 255]
    UScaled,
    /// Value will be intepreted as a signed integer, then cast to a float with the same magnitude.
    /// For example, R8_SSCALED will be converted to a float in the range of [-128, 127]
    SScaled,
    /// Value will be directly interpreted as an integer in the range of [0, 255]
    UInt,
    /// Value will be directly interpreted as an integer in the range of [-128, 127]
    SInt,

    sRGB,
    SFloat,
    UFloat,
}

#[derive(Debug, Clone)]
pub struct FormatProperties {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,

    /// Number of bytes per "block".
    pub block_size: u32,
    /// Extent of pixels per "block".
    pub block_extent: UVec2,
    pub ty: FormatType,
    pub permutation: Permutation,
}

impl FormatProperties {
    pub fn bytes_required_for_texture(&self, base_size: UVec3, mip_level: u32) -> u64 {
        let mut current_size = base_size;
        let mut total_bytes = 0;
        for _ in 0..mip_level {
            total_bytes += (current_size.x.div_ceil(self.block_extent.x)
                * current_size.y.div_ceil(self.block_extent.y)
                * current_size.z
                * self.block_size) as u64;
            current_size.x = current_size.x.div_ceil(2);
            current_size.y = current_size.y.div_ceil(2);
            current_size.z = current_size.z.div_ceil(2);
        }
        total_bytes
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Copy)]
pub enum Permutation {
    R,
    G,
    B,
    RG,
    RGB,
    BGR,
    RGBA,
    BGRA,
    ARGB,
    ABGR,

    /// A three-component format with shared exponent.
    EBGR,

    /// Depth
    D,
    /// Stencil
    S,
    /// Depth Stencil
    DS,

    /// Each 64-bit compressed texel block encodes a 4x4 rectangle of unsigned normalized RGB texel data.
    BC1_RGB,
    /// Each 64-bit compressed texel block encodes a 4x4 rectangle of unsigned normalized RGB texel data, and provides 1 bit of alpha.
    BC1_RGBA,

    BC2,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7,
    ETC2_RGB,
    ETC2_RGBA,
    EAC_R,
    EAC_RG,
    ASTC {
        x: u8,
        y: u8,
    },
}

#[derive(Clone, Debug)]
pub struct ColorSpace {
    pub primaries: ColorSpacePrimaries,
    pub transfer_function: ColorSpaceTransferFunction,
}

impl From<vk::ColorSpaceKHR> for ColorSpace {
    fn from(value: vk::ColorSpaceKHR) -> Self {
        match value {
            vk::ColorSpaceKHR::SRGB_NONLINEAR => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::DCI_P3,
                transfer_function: ColorSpaceTransferFunction::DisplayP3,
            },
            vk::ColorSpaceKHR::DISPLAY_P3_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::DCI_P3,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::scRGB,
            },
            vk::ColorSpaceKHR::DCI_P3_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::XYZ,
                transfer_function: ColorSpaceTransferFunction::DCI_P3,
            },
            vk::ColorSpaceKHR::BT709_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::BT709_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT709,
                transfer_function: ColorSpaceTransferFunction::ITU,
            },
            vk::ColorSpaceKHR::BT2020_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::HDR10_ST2084_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::ST2084_PQ,
            },
            vk::ColorSpaceKHR::HDR10_HLG_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::BT2020,
                transfer_function: ColorSpaceTransferFunction::HLG,
            },
            vk::ColorSpaceKHR::ADOBERGB_LINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::ADOBE_RGB,
                transfer_function: ColorSpaceTransferFunction::LINEAR,
            },
            vk::ColorSpaceKHR::ADOBERGB_NONLINEAR_EXT => ColorSpace {
                primaries: ColorSpacePrimaries::ADOBE_RGB,
                transfer_function: ColorSpaceTransferFunction::AdobeRGB,
            },
            _ => panic!(),
        }
    }
}

impl Format {
    #[rustfmt::skip]
    pub fn properties(&self) -> FormatProperties {
        match self {
            Format::R4G4_UNORM_PACK8 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 4, g: 4, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            Format::R4G4B4A4_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            Format::B4G4R4A4_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 4, g: 4, b: 4, a: 4, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            Format::R5G6B5_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            Format::B5G6R5_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 5, g: 6, b: 5, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR },
            Format::R5G5B5A1_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            Format::B5G5R5A1_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            Format::A1R5G5B5_UNORM_PACK16 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 5, g: 5, b: 5, a: 1, ty: FormatType::UNorm, permutation: Permutation::ARGB },

            Format::R8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R },
            Format::R8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R },
            Format::R8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R },
            Format::R8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R },
            Format::R8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            Format::R8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            Format::R8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::R },

            Format::R8G8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            Format::R8G8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG },
            Format::R8G8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG },
            Format::R8G8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG },
            Format::R8G8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            Format::R8G8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            Format::R8G8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 8, g: 8, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::RG },

            Format::R8G8B8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            Format::R8G8B8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB },
            Format::R8G8B8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB },
            Format::R8G8B8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB },
            Format::R8G8B8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            Format::R8G8B8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            Format::R8G8B8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::RGB },

            Format::B8G8R8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::BGR },
            Format::B8G8R8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SNorm, permutation: Permutation::BGR },
            Format::B8G8R8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UScaled, permutation: Permutation::BGR },
            Format::B8G8R8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SScaled, permutation: Permutation::BGR },
            Format::B8G8R8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UInt, permutation: Permutation::BGR },
            Format::B8G8R8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::SInt, permutation: Permutation::BGR },
            Format::B8G8R8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::BGR },

            Format::R8G8B8A8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            Format::R8G8B8A8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::RGBA },
            Format::R8G8B8A8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::RGBA },
            Format::R8G8B8A8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::RGBA },
            Format::R8G8B8A8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::RGBA },
            Format::R8G8B8A8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::RGBA },
            Format::R8G8B8A8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::RGBA },

            Format::B8G8R8A8_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::BGRA },
            Format::B8G8R8A8_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::BGRA },
            Format::B8G8R8A8_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::BGRA },
            Format::B8G8R8A8_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::BGRA },
            Format::B8G8R8A8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::BGRA },
            Format::B8G8R8A8_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::BGRA },
            Format::B8G8R8A8_SRGB => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::BGRA },

            Format::A8B8G8R8_UNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ABGR },
            Format::A8B8G8R8_SNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SNorm, permutation: Permutation::ABGR },
            Format::A8B8G8R8_USCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UScaled, permutation: Permutation::ABGR },
            Format::A8B8G8R8_SSCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SScaled, permutation: Permutation::ABGR },
            Format::A8B8G8R8_UINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UInt, permutation: Permutation::ABGR },
            Format::A8B8G8R8_SINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::SInt, permutation: Permutation::ABGR },
            Format::A8B8G8R8_SRGB_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ABGR },

            Format::A2R10G10B10_UNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ARGB },
            Format::A2R10G10B10_SNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ARGB },
            Format::A2R10G10B10_USCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ARGB },
            Format::A2R10G10B10_SSCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ARGB },
            Format::A2R10G10B10_UINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ARGB },
            Format::A2R10G10B10_SINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ARGB },

            Format::A2B10G10R10_UNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UNorm, permutation: Permutation::ABGR },
            Format::A2B10G10R10_SNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SNorm, permutation: Permutation::ABGR },
            Format::A2B10G10R10_USCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UScaled, permutation: Permutation::ABGR },
            Format::A2B10G10R10_SSCALED_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SScaled, permutation: Permutation::ABGR },
            Format::A2B10G10R10_UINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::UInt, permutation: Permutation::ABGR },
            Format::A2B10G10R10_SINT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 10, g: 10, b: 10, a: 2, ty: FormatType::SInt, permutation: Permutation::ABGR },

            Format::R16_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::R },
            Format::R16_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::R },
            Format::R16_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::R },
            Format::R16_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::R },
            Format::R16_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            Format::R16_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            Format::R16_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            Format::R16G16_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::RG },
            Format::R16G16_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::RG },
            Format::R16G16_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::UScaled, permutation: Permutation::RG },
            Format::R16G16_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::SScaled, permutation: Permutation::RG },
            Format::R16G16_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            Format::R16G16_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            Format::R16G16_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 16, g: 16, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            Format::R16G16B16_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::UNorm, permutation: Permutation::RGB },
            Format::R16G16B16_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::SNorm, permutation: Permutation::RGB },
            Format::R16G16B16_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::UScaled, permutation: Permutation::RGB },
            Format::R16G16B16_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::SScaled, permutation: Permutation::RGB },
            Format::R16G16B16_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            Format::R16G16B16_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            Format::R16G16B16_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 6, r: 16, g: 16, b: 16, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            Format::R16G16B16A16_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::UNorm, permutation: Permutation::RGBA },
            Format::R16G16B16A16_SNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::SNorm, permutation: Permutation::RGBA },
            Format::R16G16B16A16_USCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::UScaled, permutation: Permutation::RGBA },
            Format::R16G16B16A16_SSCALED => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::SScaled, permutation: Permutation::RGBA },
            Format::R16G16B16A16_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::UInt, permutation: Permutation::RGBA },
            Format::R16G16B16A16_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::SInt, permutation: Permutation::RGBA },
            Format::R16G16B16A16_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 16, g: 16, b: 16, a: 16, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            Format::R32_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 32, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            Format::R32_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 32, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            Format::R32_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            Format::R32G32_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 32, g: 32, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            Format::R32G32_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 32, g: 32, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            Format::R32G32_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 32, g: 32, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            Format::R32G32B32_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 12, r: 32, g: 32, b: 32, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            Format::R32G32B32_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 12, r: 32, g: 32, b: 32, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            Format::R32G32B32_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 12, r: 32, g: 32, b: 32, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            Format::R32G32B32A32_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 32, g: 32, b: 32, a: 32, ty: FormatType::UInt, permutation: Permutation::RGBA },
            Format::R32G32B32A32_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 32, g: 32, b: 32, a: 32, ty: FormatType::SInt, permutation: Permutation::RGBA },
            Format::R32G32B32A32_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 32, g: 32, b: 32, a: 32, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            Format::R64_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 64, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::R },
            Format::R64_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 64, g: 0, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::R },
            Format::R64_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 8, r: 64, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::R },

            Format::R64G64_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 64, g: 64, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::RG },
            Format::R64G64_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 64, g: 64, b: 0, a: 0, ty: FormatType::SInt, permutation: Permutation::RG },
            Format::R64G64_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 16, r: 64, g: 64, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::RG },

            Format::R64G64B64_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 24, r: 64, g: 64, b: 64, a: 0, ty: FormatType::UInt, permutation: Permutation::RGB },
            Format::R64G64B64_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 24, r: 64, g: 64, b: 64, a: 0, ty: FormatType::SInt, permutation: Permutation::RGB },
            Format::R64G64B64_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 24, r: 64, g: 64, b: 64, a: 0, ty: FormatType::SFloat, permutation: Permutation::RGB },

            Format::R64G64B64A64_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 32, r: 64, g: 64, b: 64, a: 64, ty: FormatType::UInt, permutation: Permutation::RGBA },
            Format::R64G64B64A64_SINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 32, r: 64, g: 64, b: 64, a: 64, ty: FormatType::SInt, permutation: Permutation::RGBA },
            Format::R64G64B64A64_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 32, r: 64, g: 64, b: 64, a: 64, ty: FormatType::SFloat, permutation: Permutation::RGBA },

            Format::B10G11R11_UFLOAT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 11, g: 11, b: 10, a: 0, ty: FormatType::UFloat, permutation: Permutation::BGR },
            Format::E5B9G9R9_UFLOAT_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 9, g: 9, b: 9, a: 5, ty: FormatType::UFloat, permutation: Permutation::EBGR },

            Format::D16_UNORM => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 2, r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D },
            Format::X8_D24_UNORM_PACK32 => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::D },
            Format::D32_SFLOAT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::D },
            Format::S8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 1, r: 8, g: 0, b: 0, a: 0, ty: FormatType::UInt, permutation: Permutation::S },

            Format::D16_UNORM_S8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 3, r: 16, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS },
            Format::D24_UNORM_S8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 4, r: 24, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::DS },
            Format::D32_SFLOAT_S8_UINT => FormatProperties { block_extent: UVec2 { x: 1, y: 1 }, block_size: 5, r: 32, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::DS },

            Format::BC1_RGB_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC1_RGB },
            Format::BC1_RGB_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC1_RGB },
            Format::BC1_RGBA_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC1_RGBA },
            Format::BC1_RGBA_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC1_RGBA },
            Format::BC2_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC2 },
            Format::BC2_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC2 },
            Format::BC3_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC3 },
            Format::BC3_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC3 },
            Format::BC4_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC4 },
            Format::BC4_SNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 0, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::BC4 },
            Format::BC5_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC5 },
            Format::BC5_SNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::BC5 },
            Format::BC6H_UFLOAT_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UFloat, permutation: Permutation::BC6H },
            Format::BC6H_SFLOAT_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::SFloat, permutation: Permutation::BC6H },
            Format::BC7_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::BC7 },
            Format::BC7_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::BC7 },

            Format::ETC2_R8G8B8_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 8, g: 8, b: 8, a: 0, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGB },
            Format::ETC2_R8G8B8_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 8, g: 8, b: 8, a: 0, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGB },
            Format::ETC2_R8G8B8A1_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 8, g: 8, b: 8, a: 1, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA },
            Format::ETC2_R8G8B8A1_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 8, g: 8, b: 8, a: 1, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA },
            Format::ETC2_R8G8B8A8_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 8, g: 8, b: 8, a: 8, ty: FormatType::UNorm, permutation: Permutation::ETC2_RGBA },
            Format::ETC2_R8G8B8A8_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 8, g: 8, b: 8, a: 8, ty: FormatType::sRGB, permutation: Permutation::ETC2_RGBA },

            Format::EAC_R11_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 11, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_R },
            Format::EAC_R11_SNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 8, r: 11, g: 0, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_R },
            Format::EAC_R11G11_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 11, g: 11, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::EAC_RG },
            Format::EAC_R11G11_SNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 11, g: 11, b: 0, a: 0, ty: FormatType::SNorm, permutation: Permutation::EAC_RG },

            Format::ASTC_4x4_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 4, y: 4 } },
            Format::ASTC_4x4_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 4, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 4, y: 4 } },
            Format::ASTC_5x4_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 5, y: 4 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 4 } },
            Format::ASTC_5x4_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 5, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 4 } },
            Format::ASTC_5x5_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 5, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 5, y: 5 } },
            Format::ASTC_5x5_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 5, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 5, y: 5 } },
            Format::ASTC_6x5_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 6, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 5 } },
            Format::ASTC_6x5_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 6, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 5 } },
            Format::ASTC_6x6_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 6, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 6, y: 6 } },
            Format::ASTC_6x6_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 6, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 6, y: 6 } },
            Format::ASTC_8x5_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 5 } },
            Format::ASTC_8x5_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 5 } },
            Format::ASTC_8x6_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 6 } },
            Format::ASTC_8x6_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 6 } },
            Format::ASTC_8x8_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 8 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 8, y: 8 } },
            Format::ASTC_8x8_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 8, y: 8 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 8, y: 8 } },
            Format::ASTC_10x5_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 5 } },
            Format::ASTC_10x5_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 5 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 5 } },
            Format::ASTC_10x6_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 6 } },
            Format::ASTC_10x6_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 6 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 6 } },
            Format::ASTC_10x8_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 8 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 8 } },
            Format::ASTC_10x8_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 8 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 8 } },
            Format::ASTC_10x10_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 10 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 10, y: 10 } },
            Format::ASTC_10x10_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 10, y: 10 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 10, y: 10 } },
            Format::ASTC_12x10_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 12, y: 10 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 10 } },
            Format::ASTC_12x10_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 12, y: 10 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 10 } },
            Format::ASTC_12x12_UNORM_BLOCK => FormatProperties { block_extent: UVec2 { x: 12, y: 12 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::UNorm, permutation: Permutation::ASTC { x: 12, y: 12 } },
            Format::ASTC_12x12_SRGB_BLOCK => FormatProperties { block_extent: UVec2 { x: 12, y: 12 }, block_size: 16, r: 0, g: 0, b: 0, a: 0, ty: FormatType::sRGB, permutation: Permutation::ASTC { x: 12, y: 12 } },
            _ => unimplemented!(),
        }
    }

    pub fn is_srgb_format(&self) -> bool {
        matches!(
            self,
            Format::R8_SRGB
                | Format::R8G8_SRGB
                | Format::R8G8B8_SRGB
                | Format::B8G8R8_SRGB
                | Format::R8G8B8A8_SRGB
                | Format::B8G8R8A8_SRGB
                | Format::A8B8G8R8_SRGB_PACK32
                | Format::BC1_RGB_SRGB_BLOCK
                | Format::BC1_RGBA_SRGB_BLOCK
                | Format::BC2_SRGB_BLOCK
                | Format::BC3_SRGB_BLOCK
                | Format::BC7_SRGB_BLOCK
                | Format::ETC2_R8G8B8_SRGB_BLOCK
                | Format::ETC2_R8G8B8A1_SRGB_BLOCK
                | Format::ETC2_R8G8B8A8_SRGB_BLOCK
                | Format::ASTC_4x4_SRGB_BLOCK
                | Format::ASTC_5x4_SRGB_BLOCK
                | Format::ASTC_5x5_SRGB_BLOCK
                | Format::ASTC_6x5_SRGB_BLOCK
                | Format::ASTC_6x6_SRGB_BLOCK
                | Format::ASTC_8x5_SRGB_BLOCK
                | Format::ASTC_8x6_SRGB_BLOCK
                | Format::ASTC_8x8_SRGB_BLOCK
                | Format::ASTC_10x5_SRGB_BLOCK
                | Format::ASTC_10x6_SRGB_BLOCK
                | Format::ASTC_10x8_SRGB_BLOCK
                | Format::ASTC_10x10_SRGB_BLOCK
                | Format::ASTC_12x10_SRGB_BLOCK
                | Format::ASTC_12x12_SRGB_BLOCK
                | Format::PVRTC1_2BPP_SRGB_BLOCK_IMG
                | Format::PVRTC1_4BPP_SRGB_BLOCK_IMG
                | Format::PVRTC2_2BPP_SRGB_BLOCK_IMG
                | Format::PVRTC2_4BPP_SRGB_BLOCK_IMG
        )
    }
    pub fn to_linear_format(&self) -> Format {
        match self {
            Format::R8_SRGB => Format::R8_UNORM,
            Format::R8G8_SRGB => Format::R8G8_UNORM,
            Format::R8G8B8_SRGB => Format::R8G8B8_UNORM,
            Format::B8G8R8_SRGB => Format::B8G8R8_UNORM,
            Format::R8G8B8A8_SRGB => Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8_SRGB => Format::B8G8R8A8_UNORM,
            Format::A8B8G8R8_SRGB_PACK32 => Format::A8B8G8R8_UNORM_PACK32,
            Format::BC1_RGB_SRGB_BLOCK => Format::BC1_RGB_UNORM_BLOCK,
            Format::BC1_RGBA_SRGB_BLOCK => Format::BC1_RGBA_UNORM_BLOCK,
            Format::BC2_SRGB_BLOCK => Format::BC2_UNORM_BLOCK,
            Format::BC3_SRGB_BLOCK => Format::BC3_UNORM_BLOCK,
            Format::BC7_SRGB_BLOCK => Format::BC7_UNORM_BLOCK,
            Format::ETC2_R8G8B8_SRGB_BLOCK => Format::ETC2_R8G8B8_UNORM_BLOCK,
            Format::ETC2_R8G8B8A1_SRGB_BLOCK => Format::ETC2_R8G8B8A1_UNORM_BLOCK,
            Format::ETC2_R8G8B8A8_SRGB_BLOCK => Format::ETC2_R8G8B8A8_UNORM_BLOCK,
            Format::ASTC_4x4_SRGB_BLOCK => Format::ASTC_4x4_UNORM_BLOCK,
            Format::ASTC_5x4_SRGB_BLOCK => Format::ASTC_5x4_UNORM_BLOCK,
            Format::ASTC_5x5_SRGB_BLOCK => Format::ASTC_5x5_UNORM_BLOCK,
            Format::ASTC_6x5_SRGB_BLOCK => Format::ASTC_6x5_UNORM_BLOCK,
            Format::ASTC_6x6_SRGB_BLOCK => Format::ASTC_6x6_UNORM_BLOCK,
            Format::ASTC_8x5_SRGB_BLOCK => Format::ASTC_8x5_UNORM_BLOCK,
            Format::ASTC_8x6_SRGB_BLOCK => Format::ASTC_8x6_UNORM_BLOCK,
            Format::ASTC_8x8_SRGB_BLOCK => Format::ASTC_8x8_UNORM_BLOCK,
            Format::ASTC_10x5_SRGB_BLOCK => Format::ASTC_10x5_UNORM_BLOCK,
            Format::ASTC_10x6_SRGB_BLOCK => Format::ASTC_10x6_UNORM_BLOCK,
            Format::ASTC_10x8_SRGB_BLOCK => Format::ASTC_10x8_UNORM_BLOCK,
            Format::ASTC_10x10_SRGB_BLOCK => Format::ASTC_10x10_UNORM_BLOCK,
            Format::ASTC_12x10_SRGB_BLOCK => Format::ASTC_12x10_UNORM_BLOCK,
            Format::ASTC_12x12_SRGB_BLOCK => Format::ASTC_12x12_UNORM_BLOCK,
            Format::PVRTC1_2BPP_SRGB_BLOCK_IMG => Format::PVRTC1_2BPP_UNORM_BLOCK_IMG,
            Format::PVRTC1_4BPP_SRGB_BLOCK_IMG => Format::PVRTC1_4BPP_UNORM_BLOCK_IMG,
            Format::PVRTC2_2BPP_SRGB_BLOCK_IMG => Format::PVRTC2_2BPP_UNORM_BLOCK_IMG,
            Format::PVRTC2_4BPP_SRGB_BLOCK_IMG => Format::PVRTC2_4BPP_UNORM_BLOCK_IMG,
            // If it's already a linear format, return it as-is
            _ => *self,
        }
    }
    /// Convert a UNORM format into its corrresponding SRGB format.
    pub fn to_srgb_format(&self) -> Option<Format> {
        if self.is_srgb_format() {
            return Some(*self);
        }
        let format = match self {
            Format::R8_UNORM => Format::R8_SRGB,
            Format::R8G8_UNORM => Format::R8G8_SRGB,
            Format::R8G8B8_UNORM => Format::R8G8B8_SRGB,
            Format::B8G8R8_UNORM => Format::B8G8R8_SRGB,
            Format::R8G8B8A8_UNORM => Format::R8G8B8A8_SRGB,
            Format::B8G8R8A8_UNORM => Format::B8G8R8A8_SRGB,
            Format::A8B8G8R8_UNORM_PACK32 => Format::A8B8G8R8_SRGB_PACK32,
            Format::BC1_RGB_UNORM_BLOCK => Format::BC1_RGB_SRGB_BLOCK,
            Format::BC1_RGBA_UNORM_BLOCK => Format::BC1_RGBA_SRGB_BLOCK,
            Format::BC2_UNORM_BLOCK => Format::BC2_SRGB_BLOCK,
            Format::BC3_UNORM_BLOCK => Format::BC3_SRGB_BLOCK,
            Format::BC7_UNORM_BLOCK => Format::BC7_SRGB_BLOCK,
            Format::ETC2_R8G8B8_UNORM_BLOCK => Format::ETC2_R8G8B8_SRGB_BLOCK,
            Format::ETC2_R8G8B8A1_UNORM_BLOCK => Format::ETC2_R8G8B8A1_SRGB_BLOCK,
            Format::ETC2_R8G8B8A8_UNORM_BLOCK => Format::ETC2_R8G8B8A8_SRGB_BLOCK,
            Format::ASTC_4x4_UNORM_BLOCK => Format::ASTC_4x4_SRGB_BLOCK,
            Format::ASTC_5x4_UNORM_BLOCK => Format::ASTC_5x4_SRGB_BLOCK,
            Format::ASTC_5x5_UNORM_BLOCK => Format::ASTC_5x5_SRGB_BLOCK,
            Format::ASTC_6x5_UNORM_BLOCK => Format::ASTC_6x5_SRGB_BLOCK,
            Format::ASTC_6x6_UNORM_BLOCK => Format::ASTC_6x6_SRGB_BLOCK,
            Format::ASTC_8x5_UNORM_BLOCK => Format::ASTC_8x5_SRGB_BLOCK,
            Format::ASTC_8x6_UNORM_BLOCK => Format::ASTC_8x6_SRGB_BLOCK,
            Format::ASTC_8x8_UNORM_BLOCK => Format::ASTC_8x8_SRGB_BLOCK,
            Format::ASTC_10x5_UNORM_BLOCK => Format::ASTC_10x5_SRGB_BLOCK,
            Format::ASTC_10x6_UNORM_BLOCK => Format::ASTC_10x6_SRGB_BLOCK,
            Format::ASTC_10x8_UNORM_BLOCK => Format::ASTC_10x8_SRGB_BLOCK,
            Format::ASTC_10x10_UNORM_BLOCK => Format::ASTC_10x10_SRGB_BLOCK,
            Format::ASTC_12x10_UNORM_BLOCK => Format::ASTC_12x10_SRGB_BLOCK,
            Format::ASTC_12x12_UNORM_BLOCK => Format::ASTC_12x12_SRGB_BLOCK,
            Format::PVRTC1_2BPP_UNORM_BLOCK_IMG => Format::PVRTC1_2BPP_SRGB_BLOCK_IMG,
            Format::PVRTC1_4BPP_UNORM_BLOCK_IMG => Format::PVRTC1_4BPP_SRGB_BLOCK_IMG,
            Format::PVRTC2_2BPP_UNORM_BLOCK_IMG => Format::PVRTC2_2BPP_SRGB_BLOCK_IMG,
            Format::PVRTC2_4BPP_UNORM_BLOCK_IMG => Format::PVRTC2_4BPP_SRGB_BLOCK_IMG,
            // Format doesn't have a corresponding sRGB variant.
            _ => return None,
        };
        Some(format)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColorSpacePrimaries {
    pub r: Vec2,
    pub g: Vec2,
    pub b: Vec2,
    pub white_point: Vec2,
}

pub mod white_points {
    use glam::Vec2;
    pub const D65: Vec2 = Vec2::new(0.3127, 0.3290);
    pub const D60: Vec2 = Vec2::new(0.32168, 0.33767);
    pub const E: Vec2 = Vec2::new(0.3333, 0.3333);
}
impl ColorSpacePrimaries {
    /// Primaries for [BT709](https://en.wikipedia.org/wiki/Rec._709) colorspace.
    ///
    /// It is the most common color space for LDR content.
    pub const BT709: Self = ColorSpacePrimaries {
        r: Vec2::new(0.64, 0.33),
        g: Vec2::new(0.3, 0.6),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };

    /// Primaries for [CIE 1931 XYZ](https://en.wikipedia.org/wiki/CIE_1931_color_space) colorspace.
    ///
    /// Commonly used as a bridge color space between other color spaces.
    pub const XYZ: Self = ColorSpacePrimaries {
        r: Vec2::new(1.0, 0.0),
        g: Vec2::new(0.0, 1.0),
        b: Vec2::new(0.0, 0.0),
        white_point: white_points::E,
    };
    pub const DCI_P3: Self = ColorSpacePrimaries {
        r: Vec2::new(0.68, 0.32),
        g: Vec2::new(0.265, 0.69),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };
    pub const BT2020: Self = ColorSpacePrimaries {
        r: Vec2::new(0.708, 0.292),
        g: Vec2::new(0.170, 0.797),
        b: Vec2::new(0.131, 0.046),
        white_point: white_points::D65,
    };
    pub const ADOBE_RGB: Self = ColorSpacePrimaries {
        r: Vec2::new(0.64, 0.33),
        g: Vec2::new(0.21, 0.71),
        b: Vec2::new(0.15, 0.06),
        white_point: white_points::D65,
    };

    /// Primaries used in ACES2065-1.
    /// Typically, this is the colorspace you would use to transfer images/animations between production studios.
    pub const ACES_AP0: Self = ColorSpacePrimaries {
        r: Vec2::new(0.7347, 0.2653),
        g: Vec2::new(0.0, 1.0),
        b: Vec2::new(0.0001, -0.0770),
        white_point: white_points::D60,
    };

    /// Primaries used in ACEScg.
    /// ACEScg is the recommended rendering color space for HDR games.
    pub const ACES_AP1: Self = ColorSpacePrimaries {
        r: Vec2::new(0.713, 0.293),
        g: Vec2::new(0.165, 0.830),
        b: Vec2::new(0.128, 0.044),
        white_point: white_points::D60,
    };

    pub fn area_size(&self) -> f32 {
        let a = (self.r - self.g).length();
        let b = (self.g - self.b).length();
        let c = (self.b - self.r).length();
        let s = (a + b + c) / 2.0;
        (s * (s - a) * (s - b) * (s - c)).sqrt()
    }
    #[allow(non_snake_case)]
    pub fn to_xyz(&self) -> Mat3 {
        use glam::{Vec3, Vec3A, Vec4, Vec4Swizzles};
        let x = Vec4::new(self.r.x, self.g.x, self.b.x, self.white_point.x);
        let y = Vec4::new(self.r.y, self.g.y, self.b.y, self.white_point.y);
        let X = x / y;
        let Z = (1.0 - x - y) / y;

        let mat = Mat3::from_cols(X.xyz(), Vec3::ONE, Z.xyz()).transpose();
        let white_point = Vec3A::new(X.w, 1.0, Z.w);

        let S = mat.inverse() * white_point;
        mat * Mat3::from_diagonal(S.into())
    }

    pub fn to_color_space(&self, other_color_space: &Self) -> Mat3 {
        if self == other_color_space {
            return Mat3::IDENTITY;
        }
        if self == &ColorSpacePrimaries::XYZ {
            return other_color_space.to_xyz().inverse();
        }
        if other_color_space == &ColorSpacePrimaries::XYZ {
            return self.to_xyz();
        }
        other_color_space.to_xyz().inverse() * self.to_xyz()
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpaceTransferFunction {
    LINEAR = 0,
    sRGB = 1,
    scRGB = 2,
    DCI_P3 = 3,
    DisplayP3 = 4,
    ITU = 5,
    ST2084_PQ = 6,
    HLG = 7,
    AdobeRGB = 8,
}

impl ColorSpaceTransferFunction {
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LINEAR)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_color_space_conversion() {
        let _mat = super::ColorSpacePrimaries::ACES_AP1.to_xyz();
    }
}
