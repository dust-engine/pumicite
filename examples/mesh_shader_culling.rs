use bevy::{ecs::schedule::IntoScheduleConfigs, window::PrimaryWindow};
use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use glam::{IVec2, Vec3Swizzles};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, EguiRenderSet};

const DENSITY_LEVEL: u32 = 2;

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    app.add_device_extension::<ash::khr::dynamic_rendering::Meta>().unwrap();
    app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| &mut x.dynamic_rendering).unwrap();

    app.add_device_extension::<ash::ext::mesh_shader::Meta>().unwrap();
    app.enable_feature::<vk::PhysicalDeviceMeshShaderFeaturesEXT>(|x| &mut x.mesh_shader).unwrap();
    app.enable_feature::<vk::PhysicalDeviceMeshShaderFeaturesEXT>(|x| &mut x.task_shader).unwrap();

    // Add egui plugin
    //app.add_plugins(pumicite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_systems(Startup, setup);
    //app.add_systems(EguiPrimaryContextPass, egui_ui);
    app.add_systems(
        PostUpdate, 
        mesh_shader_culling
            .in_set(DefaultRenderSet)
            //.before(EguiRenderSet)
    );

    app.run();
}

#[derive(Resource)]
struct MeshShadingPipeline {
    draw: Handle<GraphicsPipeline>,
}

fn setup(mut commands: Commands, asset_server: ResMut<AssetServer>) {
    commands.insert_resource(MeshShadingPipeline {
        draw: asset_server.load("mesh_shader_culling/mesh_shader_culling.gfx.pipeline.ron"),
    });
}

#[derive(Resource, bytemuck::Zeroable, bytemuck::NoUninit, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    cull_center: Vec2,
    cull_radius: f32,
}

fn mesh_shader_culling(
    mut swapchain_image: Single<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    pipeline: Res<MeshShadingPipeline>,
    graphics_pipelines: Res<Assets<GraphicsPipeline>>,
    //push_constants: Res<PushConstants>
) {
    let pipeline = graphics_pipelines.get(&pipeline.draw);

    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
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
        encoder.emit_barriers();

        let mut pass = encoder
            .begin_rendering()
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .color_attachment(0, |mut x| {
                x.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .clear([0.0, 0.0, 0.0, 0.0])
                    .store(true)
                    .view(current_swapchain_image.srgb_view().unwrap());
            })
            .begin();

        if let Some(pipeline) = pipeline {
            let pipeline = pass.retain(pipeline.clone().into_inner());
            pass.bind_pipeline(pipeline);
            pass.set_viewport(
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: current_swapchain_image.extent().x as f32,
                    height: current_swapchain_image.extent().y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            pass.set_scissor(
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: current_swapchain_image.extent().x,
                        height: current_swapchain_image.extent().y,
                    },
                }],
            );
            

            pass.push_constants(
                pipeline.layout(),
                vk::ShaderStageFlags::TASK_EXT,
                0,
                &bytemuck::bytes_of(&PushConstants {
                    cull_center: Vec2 { x: 2.0, y: 2.0 },
                    cull_radius: 1.0
                }),
            );

            // Dispatch mesh shading pipeline workgroups
            let n = match DENSITY_LEVEL {
                2 => 8,
                1 => 6,
                0 => 4,
                _ => 2
            };
            pass.draw_mesh_tasks(UVec3::new(n, n, 1));

            // TODO: add egui
        }
    });
}

// fn egui_ui(mut contexts: EguiContexts, mut push_constants: ResMut<PushConstants>) {

// }
