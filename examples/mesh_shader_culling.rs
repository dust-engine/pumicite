use bevy::prelude::*;
use bevy::{ecs::schedule::IntoScheduleConfigs, window::PrimaryWindow};
use bevy_pumicite::prelude::*;
use glam::{IVec2, Vec3Swizzles};
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, EguiRenderSet, egui};

const DENSITY_LEVEL: u32 = 2;

#[derive(SystemSet, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone, Copy)]
pub struct MainRenderPass;

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
    app.enable_feature::<vk::PhysicalDeviceMeshShaderFeaturesEXT>(|x| &mut x.task_shader)
        .unwrap();

    // Add egui plugin
    app.add_plugins(pumicite_egui::EguiPlugin::<With<PrimaryWindow>>::default());

    app.add_systems(Startup, setup.after(bevy_pumicite::CreateDevice));
    app.add_systems(EguiPrimaryContextPass, egui_ui);
    app.add_systems(
        PostUpdate,
        mesh_shader_culling
            .in_set(MainRenderPass)
            .before(EguiRenderSet),
    );

    app.add_render_set(MainRenderPass, start_main_render_pass);
    app.configure_sets(PostUpdate, MainRenderPass.in_set(DefaultRenderSet));
    app.configure_sets(PostUpdate, EguiRenderSet.in_set(MainRenderPass));

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
    commands.insert_resource(PushConstants {
        cull_center: Vec2 { x: -0.25, y: -0.25 },
        cull_radius: 0.75,
    });
}

#[derive(Resource, bytemuck::Zeroable, bytemuck::NoUninit, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    cull_center: Vec2,
    cull_radius: f32,
}

fn start_main_render_pass(
    mut ctx: SubmissionState,
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    ctx.record(|encoder| {
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

        encoder
            .begin_rendering()
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .color_attachment(0, |mut x| {
                x.image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .clear([0.0, 0.0, 0.0, 0.0])
                    .store(true)
                    .view(current_swapchain_image.srgb_view().unwrap());
            })
            .begin();
    });
}

fn mesh_shader_culling(
    mut state: SubmissionState,
    pipeline: Res<MeshShadingPipeline>,
    graphics_pipelines: Res<Assets<GraphicsPipeline>>,
    push_constants: Res<PushConstants>,
) {
    let Some(pipeline) = graphics_pipelines.get(&pipeline.draw) else {
        return;
    };
    let pipeline = pipeline.clone();

    state.render(move |mut pass| {
        let render_area = pass.render_area();
        let pipeline = pass.retain(pipeline.into_inner());
        pass.bind_pipeline(pipeline);
        pass.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: render_area.extent.width as f32,
                height: render_area.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        pass.set_scissor(0, &[render_area]);

        pass.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::TASK_EXT,
            0,
            bytemuck::bytes_of(&*push_constants),
        );

        // Dispatch mesh shading pipeline workgroups
        let n = match DENSITY_LEVEL {
            2 => 8,
            1 => 6,
            0 => 4,
            _ => 2,
        };
        pass.draw_mesh_tasks(UVec3::new(n, n, 1));
    });
}

fn egui_ui(mut contexts: EguiContexts, mut push_constants: ResMut<PushConstants>) {
    egui::Window::new("Mesh Shader Culling")
        .default_width(300.0)
        .show(contexts.ctx_mut().unwrap(), |ui| {
            ui.heading("Configurations:\n");
            ui.add(
                egui::Slider::new(&mut push_constants.cull_center.x, -0.5..=0.5)
                    .text("Cull Center X"),
            );
            ui.add(
                egui::Slider::new(&mut push_constants.cull_center.y, -0.5..=0.5)
                    .text("Cull Center Y"),
            );
            ui.add(
                egui::Slider::new(&mut push_constants.cull_radius, 0.0..=1.0).text("Cull Radius"),
            );
        });
}
