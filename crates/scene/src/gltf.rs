use std::{collections::BTreeMap, sync::Arc, u32};

use bevy_app::Plugin;
use bevy_asset::{
    AssetApp, AssetLoader, Assets, AsyncReadExt, Handle, LoadContext, ParseAssetPathError,
    io::AsyncSeekForwardExt,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    reflect::ReflectComponent,
    relationship::RelatedSpawner,
    world::{EntityWorldMut, FromWorld, World},
};
use bevy_pumicite::{
    DescriptorHeap,
    loader::TextureAsset,
    pumicite::{
        ash::{VkResult, vk},
        bindless::SamplerHandle,
        prelude::*,
    },
    staging::AsyncTransferGuard,
};
use bevy_reflect::Reflect;
use bevy_tasks::ConditionalSendFuture;
use bevy_transform::components::{GlobalTransform, Transform};
use bytemuck::{AnyBitPattern, NoUninit};
use glam::Vec4;

use crate::{Attribute, InstanceOf, Model, Primitive, PrimitiveKey};

trait MSFTTextureDDSExt {
    fn dds_source<'a>(&'a self, document: &'a gltf::Document) -> gltf::Image<'a>;
}
impl MSFTTextureDDSExt for gltf::Texture<'_> {
    fn dds_source<'a>(&'a self, document: &'a gltf::Document) -> gltf::Image<'a> {
        if let Some(ext) = self.extension_value("MSFT_texture_dds")
            && let Some(ext) = ext.as_object()
            && let Some(source) = ext.get("source")
            && let Some(source) = source.as_u64()
            && let Some(source) = document.images().nth(source as usize)
        {
            source
        } else {
            self.source()
        }
    }
}
pub use gltf::Semantic;
impl PrimitiveKey for gltf::Semantic {
    fn from_u64(key: u64) -> Self {
        let value = key as u32;
        let key = (key >> 32) as u32;
        match key {
            0 => gltf::Semantic::Positions,
            1 => gltf::Semantic::Normals,
            2 => gltf::Semantic::Tangents,
            3 => gltf::Semantic::Colors(value),
            4 => gltf::Semantic::TexCoords(value),
            5 => gltf::Semantic::Joints(value),
            6 => gltf::Semantic::Weights(value),
            _ => panic!("Undefined key"),
        }
    }

    fn to_u64(&self) -> u64 {
        let (key, value) = match self {
            gltf::Semantic::Extras(_) => unimplemented!(),
            gltf::Semantic::Positions => (0, 0),
            gltf::Semantic::Normals => (1, 0),
            gltf::Semantic::Tangents => (2, 0),
            gltf::Semantic::Colors(v) => (3, *v),
            gltf::Semantic::TexCoords(v) => (4, *v),
            gltf::Semantic::Joints(v) => (5, *v),
            gltf::Semantic::Weights(v) => (6, *v),
        };
        ((key as u64) << 32) | value as u64
    }
}
/// An error that occurs when loading a glTF file.
#[derive(thiserror::Error, Debug)]
pub enum GltfError {
    /// Unsupported primitive mode.
    #[error("unsupported primitive mode")]
    UnsupportedPrimitive {
        // The primitive mode.
        //mode: Mode,
    },
    /// Invalid glTF file.
    #[error("invalid glTF file: {0}")]
    Gltf(#[from] gltf::Error),

    /// Conflicting usage
    #[error("BufferView has conflicting usage with accessor {0}")]
    ConflictingUsage(usize),
    #[error("Some BufferViews are overlapping")]
    OverlappingBufferView,

    /// Binary blob is missing.
    #[error("the JSON chunk in a GLB file is missing")]
    MissingJson,
    /// Binary blob is missing.
    #[error("binary blob is missing")]
    MissingBlob,
    /// Decoding the base64 mesh data failed.
    //#[error("failed to decode base64 mesh data")]
    //Base64Decode(#[from] base64::DecodeError),
    /// Unsupported buffer format.
    #[error("unsupported buffer format")]
    BufferFormatUnsupported,
    /// The buffer URI was unable to be resolved with respect to the asset path.
    #[error("invalid buffer uri: {0}. asset path error={1}")]
    InvalidBufferUri(String, ParseAssetPathError),
    /// Invalid image mime type.
    #[error("invalid image mime type: {0}")]
    #[from(ignore)]
    InvalidImageMimeType(String),
    /// Error when loading a texture. Might be due to a disabled image file format feature.
    #[error("You may need to add the feature for the file format: {0}")]
    ImageError(#[from] image::ImageError),
    /// Error when loading a texture.
    #[error("Unknown texture color type: {0:?}")]
    UnknownTextureColorType(image::ColorType),
    /// Failed to read bytes from an asset path.
    //#[error("failed to read bytes from an asset path: {0}")]
    //ReadAssetBytesError(#[from] ReadAssetBytesError),
    /// Failed to load asset from an asset path.
    //#[error("failed to load asset from an asset path: {0}")]
    //AssetLoadError(#[from] AssetLoadError),
    /// Missing sampler for an animation.
    #[error("Missing sampler for animation {0}")]
    #[from(ignore)]
    MissingAnimationSampler(usize),
    /// Failed to generate tangents.
    //#[error("failed to generate tangents: {0}")]
    //GenerateTangentsError(#[from] bevy_mesh::GenerateTangentsError),
    /// Failed to generate morph targets.
    //#[error("failed to generate morph targets: {0}")]
    // MorphTarget(#[from] bevy_mesh::morph::MorphBuildError),
    /// Circular children in Nodes
    #[error("GLTF model must be a tree, found cycle instead at node indices: {0:?}")]
    #[from(ignore)]
    CircularChildren(String),
    /// Failed to load a file.
    #[error("failed to load file: {0}")]
    Io(#[from] std::io::Error),

    /// Vulkan API Error
    #[error("Vulkan API error: {0}")]
    Vulkan(#[from] vk::Result),

    /// Bevy asset API Error
    #[error("Bevy Asset API error: {0}")]
    AssetError(#[from] bevy_asset::ReadAssetBytesError),
}

pub struct GltfLoader {
    transfer: bevy_pumicite::staging::AsyncTransfer,
    allocator: Allocator,
    default_sampler: Arc<SamplerHandle>,
    heap: DescriptorHeap,
}
impl FromWorld for GltfLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        let heap = world.resource::<DescriptorHeap>().clone();
        let default_sampler = SamplerHandle::new(
            heap.sampler_heap().clone(),
            &vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                ..Default::default()
            },
        )
        .unwrap();
        Self {
            transfer: world
                .resource::<bevy_pumicite::staging::AsyncTransfer>()
                .clone(),
            allocator: world.resource::<Allocator>().clone(),
            default_sampler: Arc::new(default_sampler),
            heap,
        }
    }
}

struct BufferGPUView<'a> {
    view: gltf::buffer::View<'a>,
    buffer: Option<Arc<Buffer>>,
    alignment: u32,
    usage: BufferGPUViewUsage,
    /// If buffer view is for an index buffer, the offset into the per-scene index buffer in bytes,
    /// indexed by scene id.
    offset: smallvec::SmallVec<[u64; 1]>,
}
enum BufferGPUViewUsage {
    None,
    VertexBuffer,
    IndexBuffer { ty: vk::IndexType },
}
struct BufferCPUView<'a> {
    view: gltf::buffer::View<'a>,
    active: bool,
    buffer: Vec<u8>,
}

struct TextureView<'a> {
    view: gltf::image::Image<'a>,
    image: Option<Handle<TextureAsset>>,
    image_inline: Option<Image>,
    is_srgb: bool,
    active: bool,
}
impl BufferGPUView<'_> {
    pub fn accum_vertex_buffer_usage(&mut self, accessor: gltf::Accessor) -> Result<(), GltfError> {
        match self.usage {
            BufferGPUViewUsage::IndexBuffer { .. } => return Err(GltfError::OverlappingBufferView),
            _ => (),
        };
        let new_alignment = (accessor.size() as u32).next_power_of_two();
        self.alignment = self.alignment.max(new_alignment);
        self.usage = BufferGPUViewUsage::VertexBuffer;
        Ok(())
    }
    pub fn accum_index_buffer_usage(&mut self, accessor: gltf::Accessor) -> Result<(), GltfError> {
        let index_type = match accessor.size() {
            1 => vk::IndexType::UINT8_KHR,
            2 => vk::IndexType::UINT16,
            4 => vk::IndexType::UINT32,
            _ => return Err(GltfError::OverlappingBufferView),
        };
        match &mut self.usage {
            BufferGPUViewUsage::VertexBuffer => return Err(GltfError::OverlappingBufferView),
            BufferGPUViewUsage::None => {
                let mut ranges = BTreeMap::new();
                ranges.insert(
                    accessor.offset(),
                    (index_type, accessor.count() * accessor.size()),
                );
                self.usage = BufferGPUViewUsage::IndexBuffer { ty: index_type };
            }
            BufferGPUViewUsage::IndexBuffer { ty } => {
                if *ty != index_type {
                    return Err(GltfError::OverlappingBufferView);
                }
            }
        };
        Ok(())
    }
}

impl AssetLoader for GltfLoader {
    fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        async {
            let mut bytes = vec![0_u8; 5 * 4];
            reader.read_exact(&mut bytes).await?;
            let gltf = if b"glTF" == &bytes[0..4] {
                // is glb
                let document = {
                    let chunk_type = &bytes[16..20];
                    if chunk_type != b"JSON" {
                        return Err(GltfError::MissingJson);
                    }
                    let json_length: u32 = *bytemuck::from_bytes(&bytes[12..16]);
                    let mut json_data = vec![0_u8; json_length as usize];
                    reader.read_exact(&mut json_data).await?;
                    let json = gltf::json::deserialize::from_slice(&json_data)
                        .map_err(|err| gltf::Error::Deserialize(err))?;
                    gltf::Document::from_json(json)?
                };

                {
                    let mut blob_header = [0_u8; 8];
                    reader.read_exact(&mut blob_header).await?;
                    let _blob_length: u32 = *bytemuck::from_bytes(&blob_header[0..4]);
                    let blob_chunk_type = &blob_header[4..8];
                    if blob_chunk_type != b"BIN\0" {
                        return Err(GltfError::MissingBlob);
                    }
                };
                document
            } else {
                reader.read_to_end(&mut bytes).await?;

                let json = gltf::json::deserialize::from_slice(&bytes)
                    .map_err(|err| gltf::Error::Deserialize(err))?;
                gltf::Document::from_json(json)?
            };
            drop(bytes);

            let mut buffer_gpu_views: Vec<BufferGPUView> = gltf
                .views()
                .map(|view| BufferGPUView {
                    view,
                    buffer: None,
                    alignment: 1,
                    usage: BufferGPUViewUsage::None,
                    offset: smallvec::SmallVec::new(),
                })
                .collect();
            let mut buffer_cpu_views: Vec<BufferCPUView> = gltf
                .views()
                .map(|view| BufferCPUView {
                    view,
                    buffer: Vec::new(),
                    active: false,
                })
                .collect();

            let mut texture_views: Vec<TextureView> = gltf
                .images()
                .map(|view| TextureView {
                    view,
                    image: None,
                    is_srgb: false,
                    active: false,
                    image_inline: None,
                })
                .collect();

            // Mark all buffer views needed for CPU / GPU access
            {
                for image in gltf.images() {
                    match image.source() {
                        gltf::image::Source::View { view, .. } => {
                            buffer_cpu_views[view.index()].active = true;
                        }
                        _ => (),
                    }
                }
                for mesh in gltf.meshes() {
                    for primitive in mesh.primitives() {
                        if let Some(accessor) = primitive.indices() {
                            let view = &mut buffer_gpu_views[accessor
                                .view()
                                .expect("Sparse accessor not supported yet")
                                .index()];
                            view.accum_index_buffer_usage(accessor)?;
                        }
                        for (_semantic, accessor) in primitive.attributes() {
                            let view = &mut buffer_gpu_views[accessor
                                .view()
                                .expect("Sparse accessor not supported yet")
                                .index()];
                            view.accum_vertex_buffer_usage(accessor)?;
                        }
                    }
                }
                // Mark all base color images as srgb
                for material in gltf.materials() {
                    if let Some(base_color) = material.pbr_metallic_roughness().base_color_texture()
                    {
                        let image =
                            &mut texture_views[base_color.texture().dds_source(&gltf).index()];
                        image.is_srgb = true;
                        image.active = true;
                    }
                }
            }

            let supports_index_buffer_u8 = false;
            // Actually read all of those buffers
            {
                let mut allocator = self.allocator.clone();
                let mut buffers: Vec<
                    BTreeMap<
                        usize,
                        (
                            gltf::buffer::View<'_>,
                            &mut BufferCPUView,
                            &mut BufferGPUView,
                        ),
                    >,
                > = (0..gltf.buffers().len())
                    .map(|_| BTreeMap::new())
                    .collect::<Vec<_>>();
                let mut batch = self.transfer.batch().await?;
                for ((view, gpu_view), cpu_view) in gltf
                    .views()
                    .zip(buffer_gpu_views.iter_mut())
                    .zip(buffer_cpu_views.iter_mut())
                {
                    buffers[view.buffer().index()]
                        .insert(view.offset(), (view, cpu_view, gpu_view));
                }
                for (buffer, buffer_view) in gltf.buffers().zip(buffers.into_iter()) {
                    async fn read_from_reader<'a>(
                        reader: &mut dyn bevy_asset::io::Reader,
                        batch: &mut AsyncTransferGuard<'a>,
                        buffer_views: BTreeMap<
                            usize,
                            (
                                gltf::buffer::View<'_>,
                                &mut BufferCPUView<'_>,
                                &'a mut BufferGPUView<'_>,
                            ),
                        >,
                        allocator: &mut Allocator,
                        convert_u8_to_u16_index_buffer: bool,
                    ) -> Result<(), GltfError> {
                        let mut current_head: usize = 0;
                        for (_, (view, cpu_view, gpu_view)) in buffer_views.into_iter() {
                            let bytes_to_skip = view.offset() as isize - current_head as isize;
                            if bytes_to_skip > 0 {
                                reader.seek_forward(bytes_to_skip as u64).await?;
                            } else if bytes_to_skip < 0 {
                                return Err(GltfError::OverlappingBufferView);
                            }
                            current_head = view.offset() + view.length();
                            match (&gpu_view.usage, cpu_view.active) {
                                (BufferGPUViewUsage::None, false) => (),
                                (BufferGPUViewUsage::None, true) => {
                                    let data = &mut cpu_view.buffer;
                                    debug_assert!(data.is_empty());
                                    data.resize(view.length(), 0);
                                    reader.read_exact(data).await?;
                                }
                                (BufferGPUViewUsage::IndexBuffer { ty: index_type }, false) => {
                                    let should_spread_indices = convert_u8_to_u16_index_buffer
                                        && *index_type == vk::IndexType::UINT8_EXT;
                                    let size = if should_spread_indices {
                                        (gpu_view.view.length() as u64) * 2
                                    } else {
                                        gpu_view.view.length() as u64
                                    };

                                    let new_buffer = Buffer::new_upload(
                                        allocator.clone(),
                                        size,
                                        gpu_view.alignment as u64,
                                        vk::BufferUsageFlags::INDEX_BUFFER,
                                    )?;
                                    gpu_view.buffer = Some(Arc::new(new_buffer));
                                    let buffer_ref: &'a mut Buffer =
                                        Arc::get_mut(gpu_view.buffer.as_mut().unwrap()).unwrap();
                                    buffer_ref
                                        .update_contents_async::<_, GltfError>(
                                            async |slice| {
                                                if should_spread_indices {
                                                    let mut v = vec![0_u8; size as usize / 2];
                                                    reader
                                                        .read_exact(&mut v)
                                                        .await
                                                        .map_err(|x| GltfError::Io(x))?;
                                                    for (dst, src) in unsafe {
                                                        std::slice::from_raw_parts_mut(
                                                            slice.as_mut_ptr() as *mut u16,
                                                            size as usize / 2,
                                                        )
                                                    }
                                                    .iter_mut()
                                                    .zip(&v)
                                                    {
                                                        *dst = *src as u16;
                                                    }
                                                    Ok(())
                                                } else {
                                                    reader
                                                        .read_exact(&mut slice[0..view.length()])
                                                        .await
                                                        .map_err(|x| GltfError::Io(x))
                                                }
                                            },
                                            batch,
                                            allocator,
                                        )
                                        .await?;
                                }
                                (BufferGPUViewUsage::VertexBuffer, false) => {
                                    debug_assert!(gpu_view.buffer.is_none());
                                    let new_buffer = Buffer::new_upload(
                                        allocator.clone(),
                                        view.length() as u64,
                                        gpu_view.alignment as u64,
                                        // For usage, we're assuming vertex pulling.
                                        vk::BufferUsageFlags::STORAGE_BUFFER
                                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                                    )?;
                                    gpu_view.buffer = Some(Arc::new(new_buffer));
                                    let buffer_ref: &'a mut Buffer =
                                        Arc::get_mut(gpu_view.buffer.as_mut().unwrap()).unwrap();
                                    buffer_ref
                                        .update_contents_async::<_, GltfError>(
                                            async |slice| {
                                                reader
                                                    .read_exact(&mut slice[0..view.length()])
                                                    .await
                                                    .map_err(|x| GltfError::Io(x))
                                            },
                                            batch,
                                            allocator,
                                        )
                                        .await?;
                                }
                                (_, true) => {
                                    return Err(GltfError::OverlappingBufferView);
                                }
                            }
                            batch.flush().await?;
                        }
                        Ok::<(), GltfError>(())
                    }

                    match buffer.source() {
                        gltf::buffer::Source::Bin => {
                            // GLB buffer.
                            read_from_reader(
                                reader,
                                &mut batch,
                                buffer_view,
                                &mut allocator,
                                !supports_index_buffer_u8,
                            )
                            .await?;
                        }
                        gltf::buffer::Source::Uri(uri) => {
                            let buffer_path = load_context
                                .asset_path()
                                .resolve_embed(uri)
                                .map_err(|err| GltfError::InvalidBufferUri(uri.to_owned(), err))?;
                            let mut current_reader = load_context.read_asset(buffer_path).await?;
                            read_from_reader(
                                &mut current_reader,
                                &mut batch,
                                buffer_view,
                                &mut allocator,
                                !supports_index_buffer_u8,
                            )
                            .await?;
                        }
                    };
                }

                self.load_images(
                    &gltf,
                    load_context,
                    &buffer_cpu_views,
                    &mut texture_views,
                    &mut batch,
                )
                .await?;
                tracing::info!("GLTF transfer batch starting");
                batch.submit().await?; // Wait for all transfer jobs to finish
                tracing::info!("GLTF transfer batch ended");
                // put the images back
                for (texture_view, gltf_image) in texture_views.iter_mut().zip(gltf.images()) {
                    if let Some(image_inline) = texture_view.image_inline.take() {
                        texture_view.image = Some(
                            load_context.add_labeled_asset(
                                gltf_image
                                    .name()
                                    .map(|x| x.to_string())
                                    .unwrap_or_else(|| format!("Image{}", gltf_image.index())),
                                TextureAsset::new(
                                    image_inline,
                                    Some(self.heap.resource_heap().clone()),
                                )?,
                            ),
                        );
                    }
                }
            }

            let samplers = self.load_samplers(&gltf)?;
            let mut world = World::new();

            let materials = self.load_materials(&gltf, &texture_views, &samplers, &mut world);

            let meshes = self
                .load_meshes(
                    &gltf,
                    load_context,
                    &buffer_gpu_views,
                    &materials,
                    &mut world,
                )
                .await?;

            self.load_hierarchy(&gltf, load_context, meshes, &mut world);
            let scene = bevy_scene::Scene::new(world);

            Ok(scene)
        }
    }

    type Asset = bevy_scene::Scene;

    type Settings = ();

    type Error = GltfError;

    fn extensions(&self) -> &[&str] {
        &["gltf", "glb"]
    }
}

impl GltfLoader {
    fn load_images<'a>(
        &self,
        document: &gltf::Document,
        load_context: &mut LoadContext,
        buffer_views: &[BufferCPUView],
        texture_views: &'a mut [TextureView],
        batch: &mut AsyncTransferGuard<'a>,
    ) -> impl ConditionalSendFuture<Output = Result<(), <Self as AssetLoader>::Error>> {
        let mut allocator = self.allocator.clone();
        async move {
            for (gltf_image, texture_view) in document.images().zip(texture_views.iter_mut()) {
                if !texture_view.active {
                    continue;
                }
                match gltf_image.source() {
                    gltf::image::Source::View { view, mime_type } => {
                        let mut image_reader = image::ImageReader::new(std::io::Cursor::new(
                            &buffer_views[view.index()].buffer,
                        ));
                        let format = match mime_type {
                            "image/jpeg" => image::ImageFormat::Jpeg,
                            "image/png" => image::ImageFormat::Png,
                            _ => {
                                return Err(GltfError::InvalidImageMimeType(mime_type.to_string()));
                            }
                        };
                        image_reader.set_format(format);

                        let image = image_reader.decode()?;
                        use image::ColorType;
                        let format = match (image.color(), texture_view.is_srgb) {
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
                            _ => {
                                return Err(GltfError::Vulkan(
                                    vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
                                ));
                            }
                        };

                        let texture = Image::new_upload(
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
                                usage: vk::ImageUsageFlags::SAMPLED,
                                initial_layout: vk::ImageLayout::UNDEFINED,
                                ..Default::default()
                            },
                        )?;
                        texture_view.image_inline = Some(texture);
                        texture_view
                            .image_inline
                            .as_mut()
                            .unwrap()
                            .update_contents_async(
                                async |slice| {
                                    match image.color() {
                                        ColorType::Rgb8 => {
                                            for (dst, src) in slice
                                                .chunks_exact_mut(4)
                                                .zip(image.as_bytes().chunks(3))
                                            {
                                                dst[0..3].copy_from_slice(src);
                                            }
                                        }
                                        ColorType::Rgb16 => {
                                            for (dst, src) in slice
                                                .chunks_exact_mut(8)
                                                .zip(image.as_bytes().chunks(6))
                                            {
                                                dst[0..6].copy_from_slice(src);
                                            }
                                        }
                                        ColorType::Rgb32F => {
                                            for (dst, src) in slice
                                                .chunks_exact_mut(16)
                                                .zip(image.as_bytes().chunks(12))
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
                                batch,
                                &mut allocator,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            )
                            .await?;
                    }
                    gltf::image::Source::Uri { uri, .. } => {
                        let buffer_path = load_context
                            .asset_path()
                            .resolve_embed(uri)
                            .map_err(|err| GltfError::InvalidBufferUri(uri.to_owned(), err))?;
                        let handle = load_context.load::<TextureAsset>(buffer_path);
                        texture_view.image = Some(handle);
                    }
                };
            }
            Ok(())
        }
    }
    fn load_materials(
        &self,
        document: &gltf::Document,
        images: &[TextureView],
        samplers: &[Arc<SamplerHandle>],
        world: &mut World,
    ) -> Vec<Entity> {
        let materials = document.materials().map(|material| {
            let pbr = material.pbr_metallic_roughness();
            let mut material = GltfMaterial::default();
            if let Some(base_texture) = pbr.base_color_texture() {
                material.base_color = Some(Texture {
                    image: images[base_texture.texture().dds_source(document).index()]
                        .image
                        .as_ref()
                        .unwrap()
                        .clone(),
                    sampler: if let Some(sampler_index) = base_texture.texture().sampler().index() {
                        samplers[sampler_index].clone()
                    } else {
                        self.default_sampler.clone()
                    },
                });
            }
            material.base_color_factor = Vec4::from_array(pbr.base_color_factor());

            material
        });
        world.spawn_batch(materials).collect()
    }
    fn load_samplers(&self, document: &gltf::Document) -> VkResult<Vec<Arc<SamplerHandle>>> {
        document
            .samplers()
            .map(|gltf_sampler| {
                let sampler = SamplerHandle::new(
                    self.heap.sampler_heap().clone(),
                    &vk::SamplerCreateInfo {
                        mag_filter: match gltf_sampler.mag_filter() {
                            Some(gltf::texture::MagFilter::Linear) => vk::Filter::LINEAR,
                            Some(gltf::texture::MagFilter::Nearest) => vk::Filter::NEAREST,
                            None => vk::Filter::NEAREST,
                        },
                        min_filter: match gltf_sampler.min_filter() {
                            Some(gltf::texture::MinFilter::Linear)
                            | Some(gltf::texture::MinFilter::LinearMipmapLinear)
                            | Some(gltf::texture::MinFilter::LinearMipmapNearest) => {
                                vk::Filter::LINEAR
                            }
                            Some(gltf::texture::MinFilter::Nearest)
                            | Some(gltf::texture::MinFilter::NearestMipmapLinear)
                            | Some(gltf::texture::MinFilter::NearestMipmapNearest) => {
                                vk::Filter::NEAREST
                            }
                            None => vk::Filter::NEAREST,
                        },
                        mipmap_mode: match gltf_sampler.min_filter() {
                            Some(gltf::texture::MinFilter::LinearMipmapLinear)
                            | Some(gltf::texture::MinFilter::NearestMipmapLinear) => {
                                vk::SamplerMipmapMode::LINEAR
                            }
                            Some(gltf::texture::MinFilter::LinearMipmapNearest)
                            | Some(gltf::texture::MinFilter::NearestMipmapNearest) => {
                                vk::SamplerMipmapMode::NEAREST
                            }
                            _ => vk::SamplerMipmapMode::NEAREST,
                        },
                        address_mode_u: match gltf_sampler.wrap_s() {
                            gltf::texture::WrappingMode::ClampToEdge => {
                                vk::SamplerAddressMode::CLAMP_TO_EDGE
                            }
                            gltf::texture::WrappingMode::MirroredRepeat => {
                                vk::SamplerAddressMode::MIRRORED_REPEAT
                            }
                            gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
                        },
                        address_mode_v: match gltf_sampler.wrap_t() {
                            gltf::texture::WrappingMode::ClampToEdge => {
                                vk::SamplerAddressMode::CLAMP_TO_EDGE
                            }
                            gltf::texture::WrappingMode::MirroredRepeat => {
                                vk::SamplerAddressMode::MIRRORED_REPEAT
                            }
                            gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
                        },
                        ..Default::default()
                    },
                )?;
                Ok(Arc::new(sampler))
            })
            .collect::<VkResult<Vec<_>>>()
    }
    fn load_meshes(
        &self,
        document: &gltf::Document,
        _load_context: &mut bevy_asset::LoadContext,
        buffer_gpu_views: &[BufferGPUView],
        materials: &[Entity],
        world: &mut World,
    ) -> impl ConditionalSendFuture<Output = Result<Vec<Entity>, <Self as AssetLoader>::Error>>
    {
        async {
            let model_entities = document.meshes().map(|mesh| {
                let primitives = mesh.primitives().map(|primitive| {
                    let mut gltf_primitive = Primitive::default();
                    if let Some(indices) = primitive.indices()
                        && let Some(view) = indices.view()
                    {
                        let view = &buffer_gpu_views[view.index()];
                        gltf_primitive.index_buffer = view.buffer.clone();
                        gltf_primitive.index_buffer_offset = indices.offset() as u64;
                        gltf_primitive.index_count = indices.count() as u32;
                        gltf_primitive.index_type = match indices.data_type() {
                            gltf::accessor::DataType::U8 | gltf::accessor::DataType::I8 => {
                                vk::IndexType::UINT8_EXT
                            }
                            gltf::accessor::DataType::I16 | gltf::accessor::DataType::U16 => {
                                vk::IndexType::UINT16
                            }
                            gltf::accessor::DataType::U32 => vk::IndexType::UINT32,
                            gltf::accessor::DataType::F32 => panic!(),
                        };
                    }
                    gltf_primitive.topology = match primitive.mode() {
                        gltf::mesh::Mode::Points => vk::PrimitiveTopology::POINT_LIST,
                        gltf::mesh::Mode::Lines => vk::PrimitiveTopology::LINE_LIST,
                        gltf::mesh::Mode::LineLoop => unimplemented!(),
                        gltf::mesh::Mode::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
                        gltf::mesh::Mode::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
                        gltf::mesh::Mode::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
                        gltf::mesh::Mode::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
                    };
                    for (semantic, attribute) in primitive.attributes() {
                        let view = attribute.view().unwrap();
                        gltf_primitive
                            .attributes
                            .entry(semantic.to_u64())
                            .insert_entry(Attribute {
                                buffer: buffer_gpu_views[view.index()]
                                    .buffer
                                    .as_ref()
                                    .unwrap()
                                    .clone(),
                                offset: attribute.offset(),
                            });
                    }
                    if let Some(material_index) = primitive.material().index() {
                        gltf_primitive.material = materials[material_index];
                    } else {
                        todo!();
                        // Make the default material.
                    }
                    gltf_primitive
                });
                Model {
                    primitives: primitives.collect(),
                }
            });

            Ok(world.spawn_batch(model_entities).collect())
        }
    }

    fn load_hierarchy(
        &self,
        document: &gltf::Document,
        _load_context: &mut bevy_asset::LoadContext,
        models: Vec<Entity>,
        world: &mut World,
    ) {
        document.scenes().for_each(|scene| {
            let mut used_models = Vec::new();
            let mut instances = Vec::new();
            let _scene = world
                .spawn((GlobalTransform::default(), Transform::default()))
                .with_children(|spawner| {
                    self.load_node(
                        document,
                        scene.nodes(),
                        spawner,
                        &models,
                        &mut used_models,
                        &mut instances,
                    );
                })
                .insert(crate::Scene {
                    models: {
                        used_models.sort();
                        used_models
                    }, // sort for better coherency later on
                    instances: {
                        instances.sort();
                        instances
                    },
                })
                .id();
        });
    }

    fn load_node<'a>(
        &self,
        document: &gltf::Document,
        children: impl ExactSizeIterator<Item = gltf::Node<'a>>,
        spawner: &mut RelatedSpawner<ChildOf>,
        models: &[Entity],

        used_models: &mut Vec<Entity>,
        instances: &mut Vec<Entity>,
    ) {
        for child in children {
            let mut spawner = spawner.spawn_empty();
            Self::node_to_bundle(&mut spawner, &child, models);
            if child.children().len() > 0 {
                spawner.with_children(|spawner| {
                    self.load_node(
                        document,
                        child.children(),
                        spawner,
                        models,
                        used_models,
                        instances,
                    );
                });
            }
            instances.push(spawner.id());
            if let Some(mesh) = child.mesh() {
                used_models.push(models[mesh.index()]);
            }
        }
    }

    fn node_to_bundle(spawner: &mut EntityWorldMut, node: &gltf::Node, models: &[Entity]) {
        if let Some(mesh) = node.mesh() {
            spawner.insert(InstanceOf {
                model: models[mesh.index()],
            });
        }

        let transform = match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                Transform::from_matrix(glam::Mat4::from_cols_array_2d(&matrix))
            }
            gltf::scene::Transform::Decomposed {
                translation,
                rotation,
                scale,
            } => Transform {
                translation: glam::Vec3::from(translation),
                rotation: glam::Quat::from_array(rotation),
                scale: glam::Vec3::from(scale),
            },
        };
        spawner.insert((
            transform,
            bevy_transform::components::GlobalTransform::IDENTITY,
        ));
    }
}

#[derive(Clone)]
pub struct Texture {
    pub image: Handle<TextureAsset>,
    pub sampler: Arc<SamplerHandle>,
}

#[derive(Component, Reflect, Clone)]
#[reflect(opaque, Component)]
pub struct GltfMaterial {
    pub base_color: Option<Texture>,
    pub base_color_factor: Vec4,
}
impl Default for GltfMaterial {
    fn default() -> Self {
        Self {
            base_color: None,
            base_color_factor: Vec4::splat(1.0),
        }
    }
}
impl GltfMaterial {
    pub fn data(&self, textures: &Assets<TextureAsset>) -> GltfMaterialData {
        GltfMaterialData {
            base_color: self
                .base_color
                .as_ref()
                .and_then(|x| textures.get(&x.image))
                .map(|x| x.handle())
                .unwrap_or(u32::MAX),
            base_color_sampler: self
                .base_color
                .as_ref()
                .map(|x| x.sampler.id())
                .unwrap_or_default(),
            base_color_factor: self.base_color_factor.clone(),
            _padding: [0; _],
        }
    }
}

#[derive(NoUninit, AnyBitPattern, Clone, Copy)]
#[repr(C)]
pub struct GltfMaterialData {
    pub base_color_factor: Vec4,
    pub base_color: u32,
    pub base_color_sampler: u32,
    pub _padding: [u32; 2],
}

pub struct GltfPlugin;

impl Plugin for GltfPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.preregister_asset_loader::<GltfLoader>(&["glb", "gltf"])
            .register_type::<super::Scene>()
            .register_type::<super::InstanceOf>()
            .register_type::<super::Primitive>()
            .register_type::<super::Model>()
            .register_type::<super::ModelInstances>()
            .register_type::<GltfMaterial>()
            // I shouldn't need this. Work with the community to figure out why this is needed here.
            .register_type::<bevy_ecs::hierarchy::ChildOf>()
            .register_type::<bevy_transform::components::Transform>()
            .register_type::<bevy_transform::components::GlobalTransform>() // WTF??
            .register_type::<bevy_transform::components::TransformTreeChanged>()
            .register_type::<bevy_ecs::hierarchy::Children>(); // This is undue burden
    }
    fn finish(&self, app: &mut bevy_app::App) {
        app.init_asset_loader::<GltfLoader>();
    }
}
