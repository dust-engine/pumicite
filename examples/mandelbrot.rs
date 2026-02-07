use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_pumicite::PumiciteApp;
use bevy_pumicite::shader::ComputePipeline;
use bevy_pumicite::staging::UniformRingBuffer;
use bevy_pumicite::{DefaultRenderSet, RenderState, swapchain::SwapchainImage};
use pumicite::prelude::*;

const ZOOM_SPEED: f32 = 0.5;
const MOVE_SPEED: f32 = 500.0;

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE,
            ..Default::default()
        });
    app.add_device_extension::<ash::khr::push_descriptor::Meta>()
        .unwrap();
    app.add_systems(Update, get_user_input);

    app.add_systems(PostUpdate, mandelbrot_rendering.in_set(DefaultRenderSet));

    app.add_systems(Startup, setup);
    app.run();
}

#[derive(Resource)]
struct MandelbrotPipeline {
    draw: Handle<ComputePipeline>,
}

#[repr(C)]
#[derive(Resource, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct MandelbrotState {
    center: [f32; 2],
    scale: f32,
    max_iter: u32,
}

impl MandelbrotState {
    pub fn zoom(&mut self, delta: f32) {
        self.scale *= 1.0 + delta;
    }
    pub fn translate(&mut self, delta: Vec2) {
        self.center[0] += delta.x * self.scale;
        self.center[1] += delta.y * self.scale;
    }
}

fn setup(mut commands: Commands, asset_server: ResMut<AssetServer>) {
    // Create the pipeline from the .ron file
    commands.insert_resource(MandelbrotPipeline {
        draw: asset_server.load("mandelbrot/mandelbrot.comp.pipeline.ron"),
    });

    commands.insert_resource(MandelbrotState {
        center: [0.0, 0.0],
        scale: 0.005,
        max_iter: 1000,
    });
}

fn mandelbrot_rendering(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    pipeline: Res<MandelbrotPipeline>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut ring_buffer: ResMut<UniformRingBuffer>,
    mandelbrot_state: Res<MandelbrotState>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let pipeline = compute_pipelines.get(&pipeline.draw);

    state.record(|encoder| {
        // allocate new portion of ring buffer
        let mut buffer =
            ring_buffer.allocate_buffer(std::mem::size_of::<MandelbrotState>() as u64, 128);
        buffer
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(bytemuck::bytes_of(&*mandelbrot_state));
        // retain to extend lifetime
        let buffer = encoder.retain(buffer);

        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };

        let current_swapchain_image = encoder.lock(
            current_swapchain_image,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffer.vk_handle(),
            offset: buffer.offset(),
            range: buffer.size(),
        };

        let image_info = vk::DescriptorImageInfo {
            image_view: current_swapchain_image.linear_view().vk_handle(),
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        };

        let write_buffer = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: &buffer_info,
            ..Default::default()
        };

        let write_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        };

        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.memory_barrier(Access::CLEAR, Access::COMPUTE_READ);
        encoder.emit_barriers();

        if let Some(pipeline) = pipeline {
            // Use the Arc inside ComputePipeline directly, no clone or move
            let pipeline = encoder.retain(pipeline.clone().into_inner());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[write_buffer, write_image],
            );

            // Dispatch compute shader
            let (width, height) = (
                current_swapchain_image.extent().x,
                current_swapchain_image.extent().y,
            );
            let workgroups = UVec3::new(width.div_ceil(8), height.div_ceil(8), 1); // 8x8 workgroup size
            encoder.dispatch(workgroups);
        }
    });
}

fn get_user_input(
    mut mandelbrot_state: ResMut<MandelbrotState>,
    mut scroll_event: MessageReader<MouseWheel>,
    key_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    // We need to get the scroll event delta
    for event in scroll_event.read() {
        let delta = match event.unit {
            MouseScrollUnit::Line => event.y * -0.1 * ZOOM_SPEED,
            MouseScrollUnit::Pixel => event.y * -0.01 * ZOOM_SPEED,
        };

        // Adjust the center and zoom of the Mandelbrot state based on the scroll event
        mandelbrot_state.zoom(delta);
    }

    let mut x_change = 0.0;
    let mut y_change = 0.0;

    if key_input.pressed(KeyCode::ArrowLeft) || key_input.pressed(KeyCode::KeyA) {
        x_change += time.delta_secs() * -MOVE_SPEED;
    }
    if key_input.pressed(KeyCode::ArrowRight) || key_input.pressed(KeyCode::KeyD) {
        x_change += time.delta_secs() * MOVE_SPEED;
    }
    if key_input.pressed(KeyCode::ArrowUp) || key_input.pressed(KeyCode::KeyW) {
        y_change += time.delta_secs() * -MOVE_SPEED;
    }
    if key_input.pressed(KeyCode::ArrowDown) || key_input.pressed(KeyCode::KeyS) {
        y_change += time.delta_secs() * MOVE_SPEED;
    }
    mandelbrot_state.translate(Vec2 {
        x: x_change,
        y: y_change,
    });

    let mut zoom = 0.0;
    if key_input.pressed(KeyCode::KeyQ) {
        zoom += time.delta_secs() * 1.0 * ZOOM_SPEED;
    }
    if key_input.pressed(KeyCode::KeyE) {
        zoom -= time.delta_secs() * 1.0 * ZOOM_SPEED;
    }
    mandelbrot_state.zoom(zoom);
}
