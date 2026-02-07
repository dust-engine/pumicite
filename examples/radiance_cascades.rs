use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use bevy_pumicite::{
    DefaultRenderSet, RenderState, PumiciteApp, shader::ComputePipeline,
    staging::UniformRingBuffer, swapchain::SwapchainImage,
};
use pumicite::{image::FullImageView, image::Image, prelude::*};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

const BASE_RAY_COUNT: u32 = 4;
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
    app.add_plugins(pumicite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_device_extension::<ash::khr::push_descriptor::Meta>()
        .unwrap();

    app.add_systems(Update, user_input);
    app.add_systems(
        PostUpdate,
        radiance_rendering
            .in_set(DefaultRenderSet)
            .after(pumicite_egui::EguiRenderSet),
    );
    app.add_systems(Startup, setup);
    app.add_systems(EguiPrimaryContextPass, egui_ui);

    app.run();
}

#[derive(Resource)]
struct ComputePipelines {
    lines: Handle<ComputePipeline>,
    distances: Handle<ComputePipeline>,
    draw: Handle<ComputePipeline>,
}

#[repr(C)]
#[derive(Resource, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct PushConstants {
    clear_pass_num: u32, // 1 for true, 0 for false
    u_offset_cascade_pass: u32,
    texture_size: [u32; 2],
    draw_color: [u32; 3],
    last_image: u32, // 1 for true, 0 for false
    base_ray_count: u32,
    pad_: u32,
}
struct ImageResource {
    view: GPUMutex<FullImageView<Image>>,
    descriptor: ash::vk::DescriptorImageInfo,
    extent: vk::Extent3D,
    format: vk::Format,
}
#[derive(Resource)]
struct StorageImages {
    paint_image: ImageResource,
    distance_images: [ImageResource; 2],
    cascade_images: [ImageResource; 2],
}

#[repr(C)]
#[derive(Resource, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct NewLine {
    point1: [f32; 2],
    point2: [f32; 2],
    exists: u32,   // 1 for true, 0 for false
    _padding: u32, // padding
}

fn create_image_resource(
    allocator: &Allocator,
    format: vk::Format,
    extent: vk::Extent3D,
) -> ImageResource {
    let image = Image::new_private(
        allocator.clone(),
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE,
            initial_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        },
    )
    .unwrap();

    let view = GPUMutex::new(image.create_full_view().unwrap());
    let descriptor = vk::DescriptorImageInfo {
        image_view: view.vk_handle(),
        image_layout: vk::ImageLayout::GENERAL,
        sampler: vk::Sampler::null(),
    };

    ImageResource {
        view,
        descriptor,
        extent,
        format,
    }
}

fn setup(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut commands: Commands,
    asset_server: ResMut<AssetServer>,
    allocator: Res<Allocator>,
) {
    commands.insert_resource(ComputePipelines {
        lines: asset_server.load("radiance_cascades/draw_lines.comp.pipeline.ron"),
        distances: asset_server.load("radiance_cascades/distances.comp.pipeline.ron"),
        draw: asset_server.load("radiance_cascades/radiance_cascades.comp.pipeline.ron"),
    });
    commands.insert_resource(NewLine {
        exists: 0,
        point1: [0.0, 0.0],
        point2: [0.0, 0.0],
        _padding: 0,
    });
    commands.insert_resource(PushConstants {
        clear_pass_num: 0,
        u_offset_cascade_pass: 0,
        texture_size: [0, 0],
        draw_color: [0, 0, 255],
        last_image: 0,
        base_ray_count: BASE_RAY_COUNT,
        pad_: 0,
    });

    let window = windows
        .single()
        .expect("No window found; this app requires a window.");

    let mut storage_images_list = Vec::new();
    let image_format_list: Vec<vk::Format> = vec![
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
    ];

    let extent = ash::vk::Extent3D {
        width: window.physical_width(),
        height: window.physical_height(),
        depth: 1,
    };

    // 3 images
    for format in image_format_list {
        let resource = create_image_resource(&allocator, format, extent);
        storage_images_list.push(resource);
    }
    commands.insert_resource(StorageImages {
        cascade_images: [storage_images_list.remove(4), storage_images_list.remove(3)],
        distance_images: [storage_images_list.remove(2), storage_images_list.remove(1)],
        paint_image: storage_images_list.remove(0),
    });
}

fn resize_image_resource(
    img: &mut ImageResource,
    swap_extent: vk::Extent3D,
    allocator: &Allocator,
    push_constants: &mut PushConstants,
) {
    let new_img = create_image_resource(allocator, img.format, swap_extent);
    *img = new_img;
    push_constants.clear_pass_num = 1;
}

fn radiance_rendering(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    pipelines: Res<ComputePipelines>,
    compute_pipelines: Res<Assets<ComputePipeline>>,
    mut ring_buffer: ResMut<UniformRingBuffer>,
    mut storage_images: ResMut<StorageImages>,
    new_line: Res<NewLine>,
    mut push_constants: ResMut<PushConstants>,
    allocator: Res<Allocator>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let line_pipeline = compute_pipelines.get(&pipelines.lines);
    let distance_pipeline = compute_pipelines.get(&pipelines.distances);
    let draw_pipeline = compute_pipelines.get(&pipelines.draw);

    state.record(|encoder| {
        // allocate new portion of ring buffer
        let mut buffer = ring_buffer.allocate_buffer(4 * 6, 4);
        let bytes = bytemuck::bytes_of(&*new_line);
        buffer.as_slice_mut().unwrap().copy_from_slice(bytes);
        // retain to extend lifetime
        let buffer = encoder.retain(Box::new(buffer));

        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let swap_extent = vk::Extent3D {
            width: current_swapchain_image.extent().x,
            height: current_swapchain_image.extent().y,
            depth: 1,
        };
        push_constants.texture_size = [swap_extent.width, swap_extent.height];

        // resize images if needed
        if storage_images.paint_image.extent.width != swap_extent.width
            || storage_images.paint_image.extent.height != swap_extent.height
        {
            resize_image_resource(
                &mut storage_images.paint_image,
                swap_extent,
                &allocator,
                &mut push_constants,
            );
        }

        if storage_images.distance_images[0].extent.width != swap_extent.width
            || storage_images.distance_images[0].extent.height != swap_extent.height
        {
            for image in &mut storage_images.distance_images {
                resize_image_resource(image, swap_extent, &allocator, &mut push_constants);
            }
        }
        if storage_images.cascade_images[0].extent.width != swap_extent.width
            || storage_images.cascade_images[0].extent.height != swap_extent.height
        {
            for image in &mut storage_images.cascade_images {
                resize_image_resource(image, swap_extent, &allocator, &mut push_constants);
            }
        }

        let current_swapchain_image = encoder.lock(
            current_swapchain_image,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        let buffer_info = vk::DescriptorBufferInfo {
            buffer: buffer.vk_handle(),
            offset: buffer.offset(),
            range: buffer.size(),
        };

        let swapchain_image_view = match current_swapchain_image.srgb_view() {
            Some(ref view) => view.vk_handle(),
            None => {
                panic!("No SRGB view available, cannot bind image");
            }
        };

        let swapchain_image_info = vk::DescriptorImageInfo {
            image_view: swapchain_image_view,
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
        };

        let write_buffer = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buffer_info,
            ..Default::default()
        };

        let write_swapchain_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 4,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &swapchain_image_info,
            ..Default::default()
        };

        let mut paint_image_state = ResourceState::default();
        let paint_image_view = encoder.lock(
            &storage_images.paint_image.view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let write_paint_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &storage_images.paint_image.descriptor,
            ..Default::default()
        };

        let mut distance1_image_state = ResourceState::default();
        let distance1_image_view = encoder.lock(
            &storage_images.distance_images[0].view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let write_distance1_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 2,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &storage_images.distance_images[0].descriptor,
            ..Default::default()
        };

        let mut distance2_image_state = ResourceState::default();
        let distance2_image_view = encoder.lock(
            &storage_images.distance_images[1].view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let write_distance2_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 3,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &storage_images.distance_images[1].descriptor,
            ..Default::default()
        };

        let mut cascade1_image_state = ResourceState::default();
        let cascade1_image_view = encoder.lock(
            &storage_images.cascade_images[0].view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let write_cascade1_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 5,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &storage_images.cascade_images[0].descriptor,
            ..Default::default()
        };

        let mut cascade2_image_state = ResourceState::default();
        let cascade2_image_view = encoder.lock(
            &storage_images.cascade_images[1].view,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let write_cascade2_image = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_binding: 6,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &storage_images.cascade_images[1].descriptor,
            ..Default::default()
        };
        // Make the storage image available for compute writes. No spurious CLEAR->READ barrier here.
        encoder.use_image_resource(
            paint_image_view.image(),
            &mut paint_image_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );

        encoder.memory_barrier(Access::CLEAR, Access::COMPUTE_WRITE);
        encoder.emit_barriers();

        if let Some(line_pipeline) = line_pipeline {
            // Bind the draw lines shader
            let pipeline = encoder.retain(line_pipeline.clone().into_inner());
            encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
            encoder.push_descriptor_set(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[
                    write_buffer,
                    write_paint_image,
                    write_distance1_image,
                    write_swapchain_image,
                ],
            );
            encoder.push_constants(
                pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&*push_constants),
            );

            // Dispatch compute shader
            let (width, height) = (
                current_swapchain_image.extent().x,
                current_swapchain_image.extent().y,
            );
            let workgroups = UVec3::new(width.div_ceil(8), height.div_ceil(8), 1); // 8x8 workgroup size
            encoder.dispatch(workgroups);
        }

        encoder.memory_barrier(Access::COMPUTE_WRITE, Access::COMPUTE_READ);
        encoder.emit_barriers();

        encoder.use_image_resource(
            distance1_image_view.image(),
            &mut distance1_image_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            distance2_image_view.image(),
            &mut distance2_image_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );

        // DISTANCE PASSES, first pass sets uv's, last pass creates distance field
        let passes = (std::cmp::max(swap_extent.width, swap_extent.height) as f32)
            .log2()
            .ceil() as u32
            + 2;

        for i in 0..passes {
            if i == passes - 1 {
                push_constants.last_image = 1;
            } else {
                push_constants.clear_pass_num = i;
                push_constants.u_offset_cascade_pass = 1 << (passes - i - 1);
                push_constants.last_image = 0;
            }

            if let Some(distance_pipeline) = distance_pipeline {
                // Bind the draw lines shader
                let pipeline = encoder.retain(distance_pipeline.clone().into_inner());
                encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
                encoder.push_descriptor_set(
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    0,
                    &[
                        write_buffer,
                        write_paint_image,
                        write_distance1_image,
                        write_distance2_image,
                    ],
                );
                encoder.push_constants(
                    pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&*push_constants),
                );

                // Dispatch compute shader
                let (width, height) = (
                    current_swapchain_image.extent().x,
                    current_swapchain_image.extent().y,
                );
                let workgroups = UVec3::new(width.div_ceil(8), height.div_ceil(8), 1); // 8x8 workgroup size
                encoder.dispatch(workgroups);
            }

            encoder.memory_barrier(Access::COMPUTE_WRITE, Access::COMPUTE_READ);
            encoder.memory_barrier(Access::COMPUTE_READ, Access::COMPUTE_WRITE);
            encoder.emit_barriers();
        }
        push_constants.last_image = 0;

        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            cascade1_image_view.image(),
            &mut cascade1_image_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            cascade2_image_view.image(),
            &mut cascade2_image_state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        let diagonal = (((swap_extent.width * swap_extent.width)
            + (swap_extent.height * swap_extent.height)) as f32)
            .sqrt();
        let cascade_count: u32 = ((diagonal.ln() / (BASE_RAY_COUNT as f32).ln()).ceil() as u32) + 1;

        for i in (0..cascade_count).rev() {
            // last_image is used for first_image here
            if i == cascade_count - 1 {
                push_constants.last_image = 1;
            } else {
                push_constants.last_image = 0;
            }
            push_constants.u_offset_cascade_pass = i;

            if let Some(draw_pipeline) = draw_pipeline {
                // Bind the draw lines shader
                let pipeline = encoder.retain(draw_pipeline.clone().into_inner());
                encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
                encoder.push_descriptor_set(
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout(),
                    0,
                    &[
                        write_swapchain_image,
                        write_paint_image,
                        write_distance1_image,
                        write_distance2_image,
                        write_cascade1_image,
                        write_cascade2_image,
                    ],
                );
                encoder.push_constants(
                    pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&*push_constants),
                );

                // Dispatch compute shader
                let (width, height) = (
                    current_swapchain_image.extent().x,
                    current_swapchain_image.extent().y,
                );
                let workgroups = UVec3::new(width.div_ceil(8), height.div_ceil(8), 1); // 8x8 workgroup size
                encoder.dispatch(workgroups);
            }

            encoder.memory_barrier(Access::COMPUTE_WRITE, Access::COMPUTE_READ);
            encoder.memory_barrier(Access::COMPUTE_READ, Access::COMPUTE_WRITE);
            encoder.emit_barriers();
        }
        encoder.memory_barrier(Access::COMPUTE_WRITE, Access::COLOR_ATTACHMENT_WRITE);

        // Transition swapchain image from GENERAL to COLOR_ATTACHMENT for egui
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
    });
}

fn user_input(
    mut prev_pressed: Local<bool>,
    mut prev_pos: Local<[f32; 2]>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut new_line: ResMut<NewLine>,
) {
    let window = match windows.single() {
        Ok(w) => w,
        Err(_) => return,
    };
    let pressed = mouse_input.pressed(MouseButton::Left);
    let pos_opt = window.physical_cursor_position();

    if let Some(pos) = pos_opt {
        let norm_pos = [
            pos.x / window.physical_width() as f32,
            pos.y / window.physical_height() as f32,
        ];
        if pressed && *prev_pressed && *prev_pos != norm_pos {
            new_line.exists = 1;
            new_line.point1 = *prev_pos;
            new_line.point2 = norm_pos;
        } else if pressed {
            new_line.exists = 1;
            new_line.point1 = norm_pos;
            new_line.point2 = norm_pos;
        }
        *prev_pos = norm_pos;
    } else if !pressed {
        new_line.exists = 0;
    }

    *prev_pressed = pressed;
}

fn egui_ui(mut contexts: EguiContexts, mut push_constants: ResMut<PushConstants>) {
    // Reset clear_pass_num to 0 if it was clearing
    if push_constants.clear_pass_num != 0 {
        push_constants.clear_pass_num = 0;
    }

    egui::Window::new("Drawing Options").show(contexts.ctx_mut().unwrap(), |ui| {
        ui.label("Use the button below to clear the image");
        if ui.button("Clear").clicked() {
            push_constants.clear_pass_num = 1;
        }
        ui.separator();
        ui.label("Draw Color:");

        // Show a row of colored buttons (each button filled with the color).
        // Clicking a colored button sets the push-constant draw color directly.
        let current_color = push_constants.draw_color;

        // Ordered list of named colors and their RGB byte triples
        let colors: &[(&str, [u32; 3])] = &[
            ("White", [255, 255, 255]),
            ("Black", [0, 0, 0]),
            ("Red", [255, 0, 0]),
            ("Green", [0, 255, 0]),
            ("Blue", [0, 0, 255]),
            ("Yellow", [255, 255, 0]),
            ("Magenta", [255, 0, 255]),
            ("Cyan", [0, 255, 255]),
            ("Orange", [255, 165, 0]),
            ("Purple", [128, 0, 128]),
        ];

        ui.horizontal_wrapped(|ui| {
            for (_name, rgb) in colors.iter() {
                // Compute a contrasting stroke color for the border so selection is visible
                let lum = 0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32;
                let stroke_color = if lum > 160.0 {
                    egui::Color32::BLACK
                } else {
                    egui::Color32::WHITE
                };

                // Highlight the currently selected color with a thicker stroke
                let is_selected = current_color == *rgb;
                let stroke = if is_selected {
                    egui::Stroke::new(2.0, stroke_color)
                } else {
                    egui::Stroke::new(1.0, egui::Color32::from_gray(100))
                };

                let button = egui::Button::new(" ")
                    .min_size(egui::vec2(28.0, 20.0))
                    .fill(egui::Color32::from_rgb(
                        rgb[0] as u8,
                        rgb[1] as u8,
                        rgb[2] as u8,
                    ))
                    .stroke(stroke);

                let resp = ui.add(button);
                if resp.clicked() {
                    push_constants.draw_color = *rgb;
                }

                // Small spacing between buttons
                ui.add_space(6.0);
            }
        });
    });
}
