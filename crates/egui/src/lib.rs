use bevy_app::{App, Plugin, PostUpdate, PreUpdate};
use bevy_asset::{Assets, Handle};
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryFilter;
use bevy_egui::egui::TextureId;
pub use bevy_egui::*;
use bevy_pumicite::staging::{BufferInitializer, HostVisibleRingBuffer};
use bevy_pumicite::{
    DefaultRenderSet, DefaultTransferSet, RenderState, PumiciteApp, shader::GraphicsPipeline,
};
use bevy_window::PrimaryWindow;
use glam::Vec2;
use pumicite::buffer::BufferLike;
use pumicite::image::{FullImageView, ImageExt, ImageLike};
use pumicite::tracking::{Access, ResourceState};
use pumicite::{
    Allocator,
    ash::{self, vk},
    buffer::RingBufferSuballocation,
    image::Image,
    sync::GPUMutex,
};
use pumicite::{HasDevice, Sampler, debug::DebugObject, utils::AsVkHandle};
use std::alloc::Layout;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(SystemSet, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
pub struct EguiRenderSet;

pub struct EguiPlugin<Filter: QueryFilter = With<PrimaryWindow>> {
    /// Whether to convert output colors to linear color space.
    ///
    /// egui performs blending and color interpolation in sRGB color space, but depending
    /// on use case you may expect the final output values to be linear.
    ///
    /// ### `false` (preferred)
    /// Output colors directly in sRGB color space. When rendering to sRGB textures
    /// (like R8G8B8A8_SRGB), set this to false and create a linear view for the texture.
    /// This prevents the hardware from applying the sRGB OETF on color values that are
    /// already in sRGB color space.
    ///
    /// ### `true` (default)
    /// Converts output colors from sRGB to linear space in the shader. Set it to true when
    /// it's impractical to alias a sRGB texture as linear as discussed above. This may be
    /// the case when egui is running as a subpass in a render pass, and other subpasses
    /// in the render pass expects the output colors to be linear.
    pub linear_colorspace: bool,
    pub framebuffer_format: pumicite::utils::format::Format,
    pub _filter: std::marker::PhantomData<Filter>,
}
impl<Filter: QueryFilter> Default for EguiPlugin<Filter> {
    fn default() -> Self {
        Self {
            linear_colorspace: true,
            framebuffer_format: pumicite::utils::format::Format::B8G8R8A8_SRGB,
            _filter: Default::default(),
        }
    }
}

// Common behavior shared between all instances of EguiPlugin
struct EguiBasePlugin;
impl Plugin for EguiBasePlugin {
    fn build(&self, app: &mut App) {
        use bevy_asset::embedded_asset;
        embedded_asset!(app, "shaders/egui.spv");
        embedded_asset!(app, "shaders/egui.gfx.pipeline.ron");
    }
}

impl<Filter: QueryFilter + Send + Sync + 'static> Plugin for EguiPlugin<Filter> {
    fn build(&self, app: &mut App) {
        app.add_plugins((EguiBasePlugin, bevy_egui::EguiPlugin::default()));
        app.add_systems(
            PreUpdate,
            set_egui_input_viewport_scale_factor.in_set(EguiInputSet::WriteEguiEvents),
        );
        app.add_systems(
            PostUpdate,
            (
                collect_outputs::<Filter>.in_set(DefaultTransferSet),
                prepare_image::<Filter>.in_set(DefaultTransferSet),
                draw::<Filter>
                    .in_set(DefaultRenderSet)
                    .in_set(EguiRenderSet)
                    .after(collect_outputs::<Filter>)
                    .after(prepare_image::<Filter>),
            )
                .after(EguiPostUpdateSet::ProcessOutput),
        );
        app.add_device_extension::<ash::khr::dynamic_rendering::Meta>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| {
            &mut x.dynamic_rendering
        })
        .unwrap();
        app.add_device_extension::<ash::khr::push_descriptor::Meta>()
            .unwrap();

        let window = app
            .world_mut()
            .query_filtered::<Entity, Filter>()
            .iter(app.world())
            .next()
            .unwrap();
        app.world_mut().entity_mut(window).insert((
            EguiContext::default(),
            PrimaryEguiContext,
            EguiMultipassSchedule::new(EguiPrimaryContextPass),
        ));

        let mut window_to_egui_context_map =
            app.world_mut()
                .resource_mut::<bevy_egui::input::WindowToEguiContextMap>();
        window_to_egui_context_map
            .window_to_contexts
            .entry(window)
            .or_default()
            .insert(window);
        window_to_egui_context_map
            .context_to_window
            .insert(window, window);
    }

    fn finish(&self, app: &mut App) {
        use bevy_asset::load_embedded_asset;
        use bevy_pumicite::shader::ron_types::*;
        let patch = GraphicsPipelineVariant {
            color_formats: BTreeMap::from_iter([(0, self.framebuffer_format)]),
            shaders: BTreeMap::from_iter([(
                ShaderStage::Fragment,
                BTreeMap::from_iter([(
                    0,
                    SpecializationConstantType::Bool(self.linear_colorspace),
                )]),
            )]),
            ..Default::default()
        };
        let pipeline = load_embedded_asset!(app, "shaders/egui.gfx.pipeline.ron", move |x| {
            *x = patch.clone();
        });
        let resource: EguiResources<Filter> = EguiResources::new(pipeline);
        app.insert_resource(resource);
    }
}

fn set_egui_input_viewport_scale_factor(mut query: Query<(&mut EguiInput, &bevy_window::Window)>) {
    for (mut input, window) in query.iter_mut() {
        input.screen_rect = Some(bevy_egui::egui::Rect {
            min: Default::default(),
            max: bevy_egui::egui::pos2(window.resolution.size().x, window.resolution.size().y),
        });
        input
            .viewports
            .get_mut(&bevy_egui::egui::ViewportId::ROOT)
            .unwrap()
            .native_pixels_per_point = Some(window.scale_factor());
    }
}

/// Collect output from egui and copy it into a host-side buffer
/// Create textures
fn collect_outputs<Filter: QueryFilter + Send + Sync + 'static>(
    state: ResMut<EguiResources<Filter>>,
    mut uploader: BufferInitializer,
    mut ctx: RenderState,
    egui_render_output: Query<&EguiRenderOutput, Filter>,
) {
    let Ok(output) = egui_render_output.single() else {
        return;
    };
    let state = state.into_inner();
    let meshes = output.paint_jobs.iter().filter_map(
        |egui::epaint::ClippedPrimitive { primitive, .. }| match primitive {
            egui::epaint::Primitive::Mesh(mesh) => Some(mesh),
            egui::epaint::Primitive::Callback(_) => None,
        },
    );

    let mut total_indices_count: usize = 0;
    let mut total_vertices_count: usize = 0;
    for mesh in meshes.clone() {
        total_indices_count += mesh.indices.len();
        total_vertices_count += mesh.vertices.len();
    }
    if total_indices_count == 0 || total_vertices_count == 0 {
        return;
    }

    ctx.record(|encoder| {
        let index_buffer = uploader.create_preinitialized_buffer(
            encoder,
            Layout::new::<u32>().repeat(total_indices_count).unwrap().0,
            |data| {
                total_indices_count = 0;
                let dst: &mut [u32] = bytemuck::cast_slice_mut(data);
                for mesh in meshes.clone() {
                    dst[total_indices_count..(total_indices_count + mesh.indices.len())]
                        .copy_from_slice(&mesh.indices);
                    total_indices_count += mesh.indices.len();
                }
            },
        );
        let vertex_buffer = uploader.create_preinitialized_buffer(
            encoder,
            Layout::new::<egui::epaint::Vertex>()
                .repeat(total_vertices_count)
                .unwrap()
                .0,
            |data| {
                total_vertices_count = 0;
                let dst: &mut [egui::epaint::Vertex] = bytemuck::cast_slice_mut(data);
                for mesh in meshes.clone() {
                    dst[total_vertices_count..(total_vertices_count + mesh.vertices.len())]
                        .copy_from_slice(&mesh.vertices);
                    total_vertices_count += mesh.vertices.len();
                }
            },
        );
        // Wrap them in GPUMutex, then write into them using the staging buffer.
        state.vertex_buffer = Some(vertex_buffer);
        state.index_buffer = Some(index_buffer);
    });
}

#[derive(Resource)]
struct EguiResources<Filter> {
    index_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    vertex_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    samplers: HashMap<egui::TextureOptions, Arc<Sampler>>,

    /// Textures shall be presumed to be in SHADER_READ_ONLY_OPTIMAL layout
    textures: BTreeMap<u64, (GPUMutex<FullImageView<Image>>, egui::TextureOptions)>,
    marker: std::marker::PhantomData<Filter>,

    pipeline: Handle<GraphicsPipeline>,
}
impl<Filter> EguiResources<Filter> {
    fn new(pipeline: Handle<GraphicsPipeline>) -> Self {
        Self {
            index_buffer: None,
            vertex_buffer: None,
            samplers: Default::default(),
            textures: Default::default(),
            marker: Default::default(),
            pipeline,
        }
    }
}

/// Create the target images.
fn prepare_image<Filter: QueryFilter + Send + Sync + 'static>(
    mut device_buffers: ResMut<EguiResources<Filter>>,
    egui_render_output: Query<&EguiRenderOutput, Filter>,
    allocator: Res<Allocator>,
    mut ctx: RenderState,
    mut host_visible_ring_buffer: ResMut<HostVisibleRingBuffer>,
) {
    let Ok(output) = egui_render_output.single() else {
        return;
    };
    for (texture_id, image_delta) in output.textures_delta.set.iter() {
        let texture_id = match texture_id {
            TextureId::Managed(id) => *id,
            TextureId::User(_) => unimplemented!(),
        };
        let (texture, mut texture_state) = if let Some((existing_img, _)) =
            device_buffers.textures.get_mut(&texture_id)
            && existing_img.image().extent().x == image_delta.image.size()[0] as u32
            && existing_img.image().extent().y == image_delta.image.size()[1] as u32
        {
            let texture_state = ResourceState::new_with_image_layout(
                Access::FRAGMENT_SAMPLED_READ,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            (existing_img, texture_state)
        } else {
            let size = image_delta.image.size();
            let create_info = vk::ImageCreateInfo {
                // egui provides color values as alpha-premultiplied gamma values.
                // We want to read that value directly from the shader so we can do blending in gamma space.
                format: vk::Format::R8G8B8A8_UNORM,
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D {
                    width: size[0] as u32,
                    height: size[1] as u32,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let image = Image::new_private(allocator.clone(), &create_info)
                .unwrap()
                .with_name(c"EGUI asset image")
                .create_full_view()
                .unwrap();
            let image = GPUMutex::new(image);
            device_buffers
                .textures
                .insert(texture_id, (image, image_delta.options));

            device_buffers
                .samplers
                .entry(image_delta.options)
                .or_insert_with(|| {
                    fn convert_filter(filter: egui::TextureFilter) -> vk::Filter {
                        match filter {
                            egui::TextureFilter::Nearest => vk::Filter::NEAREST,
                            egui::TextureFilter::Linear => vk::Filter::LINEAR,
                        }
                    }
                    let warp = match image_delta.options.wrap_mode {
                        egui::TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        egui::TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
                        egui::TextureWrapMode::MirroredRepeat => {
                            vk::SamplerAddressMode::MIRRORED_REPEAT
                        }
                    };
                    let sampler = Sampler::new(
                        allocator.device().clone(),
                        &vk::SamplerCreateInfo {
                            min_filter: convert_filter(image_delta.options.minification),
                            mag_filter: convert_filter(image_delta.options.magnification),
                            address_mode_u: warp,
                            address_mode_v: warp,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    Arc::new(sampler)
                });
            let (texture, _) = device_buffers.textures.get_mut(&texture_id).unwrap();
            (texture, ResourceState::default())
        };

        let texel_size: u32 = 4;
        let expected_size =
            texel_size * image_delta.image.size()[0] as u32 * image_delta.image.size()[1] as u32;
        let mut host_buffer =
            host_visible_ring_buffer.allocate_buffer(expected_size as u64, texel_size as u64);
        match &image_delta.image {
            egui::epaint::ImageData::Color(image) => {
                let slice: &[u8] = bytemuck::cast_slice(image.pixels.as_slice());
                host_buffer.as_slice_mut().unwrap().copy_from_slice(slice);
            } /*
              egui::epaint::ImageData::Font(font_image) => {
                  let pixels: Vec<_> = font_image.srgba_pixels(None).collect();
                  let slice: &[u8] = bytemuck::cast_slice(pixels.as_slice());
                  dst.copy_from_slice(slice);
              }
              */
        };
        ctx.record(|encoder| {
            let image = encoder.lock(texture, vk::PipelineStageFlags2::COPY);
            let host_buffer = encoder.retain(host_buffer);

            // Transition images to TRANSFER_DST_OPTIMAL layout
            encoder.use_image_resource(
                image.image(),
                &mut texture_state,
                Access::COPY_WRITE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                0..1,
                0..1,
                true,
            );
            encoder.emit_barriers();
            encoder.copy_buffer_to_texture(
                host_buffer,
                image.image(),
                &[vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: image_delta
                        .pos
                        .map(|offset| vk::Offset3D {
                            x: offset[0] as i32,
                            y: offset[1] as i32,
                            z: 0,
                        })
                        .unwrap_or_default(),
                    image_extent: vk::Extent3D {
                        width: image_delta.image.size()[0] as u32,
                        height: image_delta.image.size()[1] as u32,
                        depth: 1,
                    },
                    ..Default::default()
                }],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            // Transition the image back into READ_ONLY_OPTIMAL layout
            encoder.use_image_resource(
                image.image(),
                &mut texture_state,
                Access::COPY_WRITE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                0..1,
                0..1,
                false,
            );
            encoder.emit_barriers();
        });
    }
}

/// Issue draw commands for egui.
fn draw<Filter: QueryFilter + Send + Sync + 'static>(
    mut state: RenderState,
    mut buffers: ResMut<EguiResources<Filter>>,
    mut egui_render_output: Query<(&EguiRenderOutput, &mut EguiContext), Filter>,

    pipeline_assets: Res<Assets<GraphicsPipeline>>,
) {
    let Some(mut index_buffer) = buffers.index_buffer.take() else {
        return;
    };
    let Some(mut vertex_buffer) = buffers.vertex_buffer.take() else {
        return;
    };
    let Some(pipeline) = pipeline_assets.get(&buffers.pipeline) else {
        return;
    };
    let pipeline = pipeline.clone();

    let Ok((output, mut egui_ctx)) = egui_render_output.single_mut() else {
        return;
    };

    state.render(move |mut pass| {
        let index_buffer = pass.lock(&mut index_buffer, vk::PipelineStageFlags2::INDEX_INPUT);
        let vertex_buffer = pass.lock(
            &mut vertex_buffer,
            vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
        );
        for (texture, _) in buffers.textures.values_mut() {
            // Lock all textures we may use
            pass.lock(texture, vk::PipelineStageFlags2::FRAGMENT_SHADER);
        }
        for sampler in buffers.samplers.values() {
            // Lock all samplers we may use
            pass.retain(sampler.clone());
        }

        let pipeline = pass.retain(pipeline.into_inner());

        pass.bind_pipeline(pipeline);

        pass.bind_vertex_buffers(0, [vertex_buffer].into_iter());
        pass.bind_index_buffer(index_buffer, 0, vk::IndexType::UINT32);

        let rect = egui_ctx.get_mut().screen_rect();
        let viewport_logical_size = Vec2::new(rect.max.x - rect.min.x, rect.max.y - rect.min.y);
        let scale_factor = egui_ctx.get_mut().pixels_per_point();
        let viewport_physical_size = viewport_logical_size * scale_factor;
        pass.set_viewport(
            0,
            &[vk::Viewport {
                x: rect.min.x,
                y: rect.min.y,
                width: viewport_physical_size.x,
                height: viewport_physical_size.y,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        pass.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::cast_slice(&[viewport_logical_size.x, viewport_logical_size.y]),
        );

        let mut current_vertex = 0;
        let mut current_indice = 0;
        for egui::epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in output.paint_jobs.iter()
        {
            let mesh = match primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh,
                egui::epaint::Primitive::Callback(_) => panic!(),
            };
            let clip_min = Vec2::new(clip_rect.min.x, clip_rect.min.y) * scale_factor;
            let clip_max = Vec2::new(clip_rect.max.x, clip_rect.max.y) * scale_factor;
            let clip_extent = clip_max - clip_min;
            pass.set_scissor(
                0,
                &[vk::Rect2D {
                    extent: vk::Extent2D {
                        width: clip_extent.x.round() as u32,
                        height: clip_extent.y.round() as u32,
                    },
                    offset: vk::Offset2D {
                        x: clip_min.x.round() as i32,
                        y: clip_min.y.round() as i32,
                    },
                }],
            );
            let texture_id = match mesh.texture_id {
                TextureId::Managed(id) => id,
                TextureId::User(_) => unimplemented!(),
            };
            let (texture, options) = buffers.textures.get(&texture_id).unwrap();
            let sampler = buffers.samplers.get(options).unwrap();

            pass.push_descriptor_set(
                pipeline.layout(),
                0,
                &[vk::WriteDescriptorSet {
                    dst_binding: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }
                .image_info(&[vk::DescriptorImageInfo {
                    image_view: texture.vk_handle(),
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    sampler: sampler.vk_handle(),
                }])],
            );

            pass.draw_indexed(
                current_indice..(mesh.indices.len() as u32 + current_indice),
                0..1,
                current_vertex as i32,
            );
            current_vertex += mesh.vertices.len() as u32;
            current_indice += mesh.indices.len() as u32;
        }
    });
}
