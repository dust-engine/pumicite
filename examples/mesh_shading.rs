use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use glam::{IVec2, Vec3Swizzles};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    app.add_device_extension::<ash::khr::dynamic_rendering::Meta>()
        .unwrap();
    app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| &mut x.dynamic_rendering)
        .unwrap();
    app.add_device_extension::<ash::ext::mesh_shader::Meta>()
        .unwrap();
    app.enable_feature::<vk::PhysicalDeviceMeshShaderFeaturesEXT>(|x| &mut x.mesh_shader)
        .unwrap();
    app.add_systems(PostUpdate, mesh_shading.in_set(DefaultRenderSet));

    app.add_systems(Startup, setup);
    app.run();
}

#[derive(Resource)]
struct MeshShadingPipeline {
    draw: Handle<GraphicsPipeline>,
}

fn setup(mut commands: Commands, asset_server: ResMut<AssetServer>) {
    commands.insert_resource(MeshShadingPipeline {
        draw: asset_server.load("mesh_shading/mesh_shading.gfx.pipeline.ron"),
    });
}

fn mesh_shading(
    mut swapchain_image: Single<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    pipeline: Res<MeshShadingPipeline>,
    graphics_pipelines: Res<Assets<GraphicsPipeline>>
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

            // Bind the vertex buffer and draw the triangle
            pass.draw_mesh_tasks(UVec3::new(1, 1, 1));
        }
    });
}
