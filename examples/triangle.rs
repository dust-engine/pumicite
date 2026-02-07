use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy_pumicite::PumiciteApp;
use bevy_pumicite::shader::GraphicsPipeline;
use bevy_pumicite::staging::DeviceLocalRingBuffer;
use bevy_pumicite::{DefaultRenderSet, RenderState, swapchain::SwapchainImage};
use glam::{IVec2, Vec3Swizzles};
use pumicite::prelude::*;
fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    app.add_device_extension::<ash::khr::dynamic_rendering::Meta>()
        .unwrap();
    app.enable_feature::<vk::PhysicalDeviceDynamicRenderingFeatures>(|x| &mut x.dynamic_rendering)
        .unwrap();
    app.add_systems(PostUpdate, triangle_rendering.in_set(DefaultRenderSet));

    app.add_systems(Startup, setup);
    app.run();
}

#[derive(Resource)]
struct TrianglePipeline {
    draw: Handle<GraphicsPipeline>,
}

fn setup(mut commands: Commands, asset_server: ResMut<AssetServer>) {
    commands.insert_resource(TrianglePipeline {
        draw: asset_server.load("triangle/triangle.gfx.pipeline.ron"),
    });
}

fn triangle_rendering(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    pipeline: Res<TrianglePipeline>,
    graphics_pipelines: Res<Assets<GraphicsPipeline>>,
    mut ring_buffer: ResMut<DeviceLocalRingBuffer>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let pipeline = graphics_pipelines.get(&pipeline.draw);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
    struct Vertex {
        position: Vec3,
        color: Vec3,
    }
    let vertices = [
        Vertex {
            position: Vec3::new(0.0, -0.5, 0.0), // Bottom vertex
            color: Vec3::new(1.0, 0.0, 0.0),     // Red
        },
        Vertex {
            position: Vec3::new(-0.5, 0.5, 0.0), // Top-left vertex
            color: Vec3::new(0.0, 1.0, 0.0),     // Green
        },
        Vertex {
            position: Vec3::new(0.5, 0.5, 0.0), // Top-right vertex
            color: Vec3::new(0.0, 0.0, 1.0),    // Blue
        },
    ];

    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let buffer = ring_buffer.allocate_buffer(24 * 3, 4);
        let buffer = encoder.retain(buffer);
        encoder.update_buffer(buffer, bytemuck::cast_slice(&vertices));

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
        encoder.memory_barrier(Access::CLEAR, Access::VERTEX_READ); // guard against the vertex buffer transfer
        encoder.emit_barriers();

        let mut pass = encoder
            .begin_rendering()
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .color_attachment(0, |mut x| {
                x.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .clear([1.0, 0.0, 0.0, 0.0])
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
            pass.bind_vertex_buffers(0, [buffer].into_iter());
            pass.draw(0..3, 0..1);
        }
    });
}
