use ash::vk;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_pumicite::{DefaultRenderSet, RenderState, swapchain::SwapchainImage};
use pumicite::prelude::*;
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins)
        .add_plugins(pumicite_egui::EguiPlugin::<With<PrimaryWindow>> {
            linear_colorspace: false,
            framebuffer_format: pumicite::utils::format::Format::B8G8R8A8_UNORM,
            ..Default::default()
        });

    app.add_systems(EguiPrimaryContextPass, ui_example_system);

    app.add_systems(
        PostUpdate,
        start_main_render_pass
            .before(pumicite_egui::EguiRenderSet)
            .in_set(DefaultRenderSet),
    );

    app.run();
}

#[derive(Default)]
struct UIState {
    name: String,
    age: u32,
}
fn ui_example_system(mut contexts: EguiContexts, mut state: Local<UIState>) {
    egui::Window::new("hello").show(contexts.ctx_mut().unwrap(), |ui| {
        ui.heading("My egui Application");
        ui.horizontal(|ui| {
            let name_label = ui.label("Your name: ");
            ui.text_edit_singleline(&mut state.name)
                .labelled_by(name_label.id);
        });
        ui.add(egui::Slider::new(&mut state.age, 0..=120).text("age"));
        if ui.button("Increment").clicked() {
            state.age += 1;
        }
        ui.label(format!("Hello '{}', age {}", state.name, state.age));
    });
}

/// For egui, you are always responsible for setting up the render pass. This
/// is so that egui can piggy-back on an already existing render pass and doesn't
/// have to start a new one - important for mobile performance.
fn start_main_render_pass(
    mut ctx: RenderState,
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
            .color_attachment(0, |mut builder| {
                builder
                    .clear(Vec4::new(0.0, 0.0, 0.0, 1.0))
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .store(true)
                    .view(current_swapchain_image.linear_view());
                // Use linear view for egui. egui does all the interpolation in srgb space.
            })
            .render_area(IVec2::ZERO, current_swapchain_image.extent().xy())
            .begin();
    });
}
