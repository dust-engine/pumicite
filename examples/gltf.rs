use std::{alloc::Layout, collections::BTreeMap};

use ash::VkResult;
use bevy::{
    app::prelude::*,
    asset::prelude::*,
    ecs::prelude::*,
    transform::components::{GlobalTransform, Transform},
};
use bevy_pumicite::{loader::TextureAsset, prelude::*};
use bytemuck::{AnyBitPattern, NoUninit};
use glam::{IVec2, Mat4, UVec2, Vec2, Vec3, Vec3Swizzles, Vec4};
use pumicite::{
    buffer::RingBufferSuballocation,
    image::{FullImageView, Image},
};
use pumicite_scene::gltf::{self, GltfMaterialData};
use std::mem::size_of;

mod flycam;
use crate::flycam::{FlyCamera, FlyCameraPlugin};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins)
        .add_plugins(pumicite_scene::gltf::GltfPlugin)
        .add_plugins(FlyCameraPlugin);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut().entity_mut(primary_window).insert((
        SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        },
        GBuffer::default(),
    ));

    app.world_mut().spawn((
        GlobalTransform::default(),
        Transform::from_translation(Vec3::new(0.0, 0.5, 2.0)),
        FlyCamera {
            max_speed: 1.0,
            accel: 1.0,
            friction: 1.0,
            ..Default::default()
        },
    ));

    app.enable_feature::<vk::PhysicalDeviceFeatures>(|x| &mut x.shader_int64)
        .unwrap();
    app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| &mut x.dynamic_rendering)
        .unwrap();
    app.enable_feature::<vk::PhysicalDeviceShaderDrawParameterFeatures>(|x| {
        &mut x.shader_draw_parameters
    })
    .unwrap();

    app.enable_bindless().unwrap();

    app.add_systems(
        PostUpdate,
        (
            gbuffer_resize
                .in_set(DefaultRenderSet)
                .before(start_main_render_pass),
            start_main_render_pass
                .in_set(DefaultRenderSet)
                .before(MainRenderPass),
            prepare_gltf_scene
                .in_set(DefaultRenderSet)
                .before(start_main_render_pass),
            draw_gltf_scene
                .in_set(MainRenderPass)
                .in_set(DefaultRenderSet),
        ),
    );
    app.add_systems(Startup, setup);
    app.run();
}

#[derive(Resource)]
struct PbrPipeline {
    draw: Handle<GraphicsPipeline>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle: Handle<bevy::scene::Scene> = asset_server.load("gltf/FlightHelmet.gltf");
    commands.spawn(bevy::scene::SceneRoot(handle));
    commands.insert_resource(PbrPipeline {
        draw: asset_server.load("gltf/pbr.gfx.pipeline.ron"),
    });

    commands.insert_resource(PreparedGltfScene::default());
}

#[derive(Component, Default)]
struct GBuffer {
    textures: Option<GPUMutex<GBufferTextures>>,
    state: ResourceState,
}
struct GBufferTextures {
    depth: FullImageView<Image>,
}
impl GBufferTextures {
    pub fn new(allocator: Allocator, extent: UVec2) -> VkResult<Self> {
        let depth = Image::new_private(
            allocator,
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::D32_SFLOAT,
                extent: vk::Extent3D {
                    width: extent.x,
                    height: extent.y,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
        )?;
        Ok(Self {
            depth: depth.create_full_view()?,
        })
    }
}
fn gbuffer_resize(mut query: Query<(&SwapchainImage, &mut GBuffer)>, allocator: Res<Allocator>) {
    for (swapchain_image, mut g_buffer) in query.iter_mut() {
        let Some(swapchain_image) = swapchain_image.current_image() else {
            continue;
        };
        let g_buffer_extent = g_buffer
            .textures
            .as_ref()
            .map(|x| x.depth.image().extent())
            .unwrap_or_default();
        if g_buffer_extent != swapchain_image.extent() {
            g_buffer.textures = Some(GPUMutex::new(
                GBufferTextures::new(
                    allocator.clone(),
                    UVec2 {
                        x: swapchain_image.extent().x,
                        y: swapchain_image.extent().y,
                    },
                )
                .unwrap(),
            ));
            g_buffer.state = ResourceState::default();
        }
    }
}

#[derive(NoUninit, Clone, Copy, AnyBitPattern, Debug)]
#[repr(C)]
struct ModelData {
    p_positions: u64,
    p_normals: u64,
    p_colors: u64,
    p_texcoords: u64,
}

#[derive(NoUninit, Clone, Copy, AnyBitPattern)]
#[repr(C)]
struct CameraUniforms {
    view: Mat4,
    projection: Mat4,
}

#[derive(SystemSet, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub struct MainRenderPass;

fn start_main_render_pass(
    mut ctx: RenderState,
    mut swapchain_image: Query<
        (&mut SwapchainImage, &mut GBuffer),
        With<bevy::window::PrimaryWindow>,
    >,
) {
    let Ok((mut swapchain_image, mut gbuffer)) = swapchain_image.single_mut() else {
        return;
    };
    ctx.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let current_gbuffer = encoder.lock(
            gbuffer.textures.as_ref().unwrap(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );
        let current_swapchain_image = encoder.lock(
            current_swapchain_image,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            current_gbuffer.depth.image(),
            &mut gbuffer.state,
            Access::EARLY_FRAGMENT_TEST_WRITE,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder
            .begin_rendering()
            .color_attachment(0, |mut builder| {
                builder
                    .clear(Vec4::new(0.0, 0.0, 0.0, 1.0))
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(current_swapchain_image.srgb_view().unwrap());
            })
            .depth_attachment(|mut builder| {
                builder
                    .clear(0.0)
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(&current_gbuffer.depth);
            })
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .begin();
    });
}

#[derive(Default, Resource)]
struct PreparedGltfScene {
    material_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    primitive_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    instance_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    camera_buffer: Option<GPUMutex<RingBufferSuballocation>>,
    material_mapping: BTreeMap<Entity, usize>,
}

fn prepare_gltf_scene(
    cameras: Query<(&FlyCamera, &Transform, &GlobalTransform)>,
    mut ring_buffer: BufferInitializer,
    models: Query<(&pumicite_scene::Model, &pumicite_scene::ModelInstances)>,
    materials: Query<(Entity, &pumicite_scene::gltf::GltfMaterial)>,
    textures: Res<Assets<TextureAsset>>,
    instance_query: Query<(&GlobalTransform, &Transform, &pumicite_scene::InstanceOf)>,
    mut ctx: RenderState,
    mut prepared_scene: ResMut<PreparedGltfScene>,
    mut swapchain_image: Query<
        (&mut SwapchainImage, &mut GBuffer),
        With<bevy::window::PrimaryWindow>,
    >,
) {
    let Some((_, _, camera_transform)) = cameras.single().ok() else {
        return;
    };
    let Ok((swapchain_image, _gbuffer)) = swapchain_image.single_mut() else {
        return;
    };

    ctx.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let camera = ring_buffer.create_preinitialized_buffer(
            encoder,
            Layout::new::<CameraUniforms>(),
            |dst| {
                let camera: &mut CameraUniforms = bytemuck::from_bytes_mut(dst);
                camera.view = camera_transform.to_matrix().inverse();
                camera.projection = Mat4::perspective_infinite_reverse_rh(
                    std::f32::consts::FRAC_PI_3,
                    current_swapchain_image.extent().x as f32
                        / current_swapchain_image.extent().y as f32,
                    0.1,
                );
            },
        );

        let num_materials = materials.iter().len();
        let mut material_mapping: BTreeMap<Entity, usize> = BTreeMap::new();
        let material_buffer = ring_buffer.create_preinitialized_buffer(
            encoder,
            std::alloc::Layout::new::<GltfMaterialData>()
                .repeat(num_materials)
                .unwrap()
                .0,
            |dst| {
                let dst: &mut [GltfMaterialData] = bytemuck::cast_slice_mut(dst);
                for (index, ((entity, material), dst)) in materials.iter().zip(dst).enumerate() {
                    material_mapping.insert(entity, index);
                    *dst = GltfMaterialData {
                        base_color_factor: material.base_color_factor,
                        base_color: material
                            .base_color
                            .as_ref()
                            .and_then(|x| textures.get(&x.image))
                            .map(|x| x.handle())
                            .unwrap_or(u32::MAX),
                        base_color_sampler: material
                            .base_color
                            .as_ref()
                            .map(|x| x.sampler.id())
                            .unwrap_or(u32::MAX),
                        _padding: [0; 2],
                    };
                }
            },
        );

        // Calculate the total size of the model data.
        let num_primitives: usize = models.iter().map(|x| x.0.len()).sum();
        let primitive_buffer = ring_buffer.create_preinitialized_buffer(
            encoder,
            std::alloc::Layout::new::<ModelData>()
                .repeat(num_primitives)
                .unwrap()
                .0,
            |dst| {
                let dst: &mut [ModelData] = bytemuck::cast_slice_mut(dst);
                for (primitive, dst) in models.iter().flat_map(|(model, _)| model.iter()).zip(dst) {
                    *dst = ModelData {
                        p_positions: primitive.attribute_gpuva(gltf::Semantic::Positions),
                        p_normals: primitive.attribute_gpuva(gltf::Semantic::Normals),
                        p_colors: primitive.attribute_gpuva(gltf::Semantic::Colors(0)),
                        p_texcoords: primitive.attribute_gpuva(gltf::Semantic::TexCoords(0)),
                    };
                }
            },
        );
        let num_instances: usize = models.iter().map(|(_, instances)| instances.len()).sum();
        let instance_buffer = ring_buffer.create_preinitialized_buffer(
            encoder,
            std::alloc::Layout::new::<Mat4>()
                .repeat_packed(num_instances)
                .unwrap(),
            |dst| {
                let dst: &mut [Mat4] = bytemuck::cast_slice_mut(dst);
                for (instance, dst) in models
                    .iter()
                    .flat_map(|(_, instances)| instances.iter())
                    .zip(dst)
                {
                    let (global_transform, _, _) = instance_query.get(instance).unwrap();
                    *dst = global_transform.to_matrix();
                }
            },
        );
        prepared_scene.instance_buffer = Some(instance_buffer);
        prepared_scene.material_buffer = Some(material_buffer);
        prepared_scene.primitive_buffer = Some(primitive_buffer);
        prepared_scene.camera_buffer = Some(camera);
        prepared_scene.material_mapping = material_mapping;
    });
}

fn draw_gltf_scene(
    pbr_pipeline: Res<PbrPipeline>,
    models: Query<(&pumicite_scene::Model, &pumicite_scene::ModelInstances)>,
    mut ctx: RenderState,

    pipelines: Res<Assets<GraphicsPipeline>>,

    mut swapchain_image: Query<
        (&mut SwapchainImage, &mut GBuffer),
        With<bevy::window::PrimaryWindow>,
    >,
    heap: Res<DescriptorHeap>,
    mut prepared_scene: ResMut<PreparedGltfScene>,
) {
    if models.is_empty() {
        return;
    }
    let Some(pipeline) = pipelines.get(&pbr_pipeline.draw) else {
        return;
    };
    let Ok((swapchain_image, _gbuffer)) = swapchain_image.single_mut() else {
        return;
    };

    ctx.render(|mut render_pass| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let (
            Some(primitive_buffer),
            Some(instance_buffer),
            Some(material_buffer),
            Some(camera_buffer),
        ) = (
            prepared_scene.primitive_buffer.as_ref(),
            prepared_scene.instance_buffer.as_ref(),
            prepared_scene.material_buffer.as_ref(),
            prepared_scene.camera_buffer.as_ref(),
        )
        else {
            return;
        };

        heap.bind(&mut *render_pass, vk::PipelineBindPoint::GRAPHICS);
        let primitive_buffer =
            render_pass.lock(primitive_buffer, vk::PipelineStageFlags2::VERTEX_SHADER);
        let camera_buffer = render_pass.lock(camera_buffer, vk::PipelineStageFlags2::VERTEX_SHADER);
        let material_buffer =
            render_pass.lock(material_buffer, vk::PipelineStageFlags2::VERTEX_SHADER);
        let instance_buffer =
            render_pass.lock(instance_buffer, vk::PipelineStageFlags2::VERTEX_SHADER);
        let material_mapping = std::mem::take(&mut prepared_scene.material_mapping);
        let pipeline = render_pass.retain(pipeline.clone().into_inner());
        render_pass.bind_pipeline(pipeline);
        let viewport_physical_size = Vec2::new(
            current_swapchain_image.extent().x as f32,
            current_swapchain_image.extent().y as f32,
        );
        render_pass.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: viewport_physical_size.y,
                width: viewport_physical_size.x,
                height: -viewport_physical_size.y,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        render_pass.set_scissor(
            0,
            &[vk::Rect2D {
                extent: vk::Extent2D {
                    width: current_swapchain_image.extent().x,
                    height: current_swapchain_image.extent().y,
                },
                ..Default::default()
            }],
        );

        let mut current_primitive = 0;
        let mut current_instance = 0;
        for (model, instances) in models.iter() {
            for primitive in model.iter() {
                if primitive.topology != vk::PrimitiveTopology::TRIANGLE_LIST {
                    continue;
                }

                let push_constants = PushConstants {
                    primitive_buffer_gpuva: primitive_buffer.device_address()
                        + current_primitive * size_of::<ModelData>() as u64,
                    instance_buffer_gpuva: instance_buffer.device_address()
                        + current_instance * size_of::<Mat4>() as u64,
                    camera_buffer_gpuva: camera_buffer.device_address(),
                    material_buffer_gpuva: material_buffer.device_address()
                        + *material_mapping.get(&primitive.material).unwrap() as u64
                            * size_of::<GltfMaterialData>() as u64,
                };
                render_pass.push_constants(
                    pipeline.layout(),
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                if let Some(index_buffer) = &primitive.index_buffer {
                    let index_buffer = render_pass.retain(index_buffer.clone());
                    render_pass.bind_index_buffer(
                        index_buffer.as_ref(),
                        primitive.index_buffer_offset as u64,
                        primitive.index_type,
                    );
                    render_pass.draw_indexed(
                        0..primitive.index_count,
                        0..instances.len() as u32,
                        0,
                    );
                } else {
                    // render_pass.draw(primi, instance_count, first_vertex, first_instance);
                }
                current_primitive += 1;
            }
            current_instance += instances.len() as u64;
        }
    });
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    primitive_buffer_gpuva: u64,
    instance_buffer_gpuva: u64,
    camera_buffer_gpuva: u64,
    material_buffer_gpuva: u64,
}
