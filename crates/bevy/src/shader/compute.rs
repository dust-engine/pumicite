use std::{borrow::Cow, ffi::CString, ops::Deref, sync::Arc};

use bevy_asset::{Asset, AssetLoader, LoadedAsset};
use bevy_ecs::world::FromWorld;
use bevy_reflect::TypePath;
use pumicite::{
    HasDevice,
    ash::vk,
    pipeline::{Pipeline, PipelineCache, ShaderEntry},
};

#[cfg(any(feature = "ron", feature = "postcard"))]
use crate::{
    DescriptorHeap,
    shader::{PipelineLayoutLoader, PipelineLoaderError, ShaderModule},
};

/// A loaded compute pipeline asset.
///
/// Load via asset server with `.comp.pipeline.ron` extension.
#[derive(Clone, Asset, TypePath)]
pub struct ComputePipeline(Arc<Pipeline>);

impl ComputePipeline {
    pub fn into_inner(self) -> Arc<Pipeline> {
        self.0
    }
}

/// Asset loader for compute pipelines (`.comp.pipeline.ron` files).
///
/// Loads compute pipeline configurations and creates Vulkan compute pipelines.
#[cfg(any(feature = "ron", feature = "postcard"))]
pub struct ComputePipelineLoader {
    pipeline_cache: Arc<PipelineCache>,
    heap: Option<DescriptorHeap>,
}
#[cfg(any(feature = "ron", feature = "postcard"))]
impl FromWorld for ComputePipelineLoader {
    fn from_world(world: &mut bevy_ecs::world::World) -> Self {
        Self {
            pipeline_cache: world
                .resource::<pumicite::bevy::PipelineCache>()
                .deref()
                .clone(),
            heap: world.get_resource().cloned(),
        }
    }
}

#[cfg(any(feature = "ron", feature = "postcard"))]
impl AssetLoader for ComputePipelineLoader {
    type Asset = ComputePipeline;
    type Settings = ();
    type Error = PipelineLoaderError;

    async fn load(
        &self,
        reader: &mut dyn bevy_asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy_asset::LoadContext<'_>,
    ) -> Result<ComputePipeline, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let ext = load_context
            .asset_path()
            .get_full_extension()
            .unwrap_or_default();
        let pipeline: pumicite_types::ComputePipeline = super::deserialize(&bytes, &ext)?;

        let layout = match &pipeline.layout {
            pumicite_types::PipelineLayoutRef::Inline(pipeline_layout) => {
                PipelineLayoutLoader::load_inner(
                    pipeline_layout,
                    self.pipeline_cache.device().clone(),
                    self.heap.as_ref(),
                    load_context,
                )
                .await?
                .0
            }
            pumicite_types::PipelineLayoutRef::Path(path) => {
                load_context
                    .loader()
                    .immediate()
                    .load::<pumicite::bevy::PipelineLayout>(path)
                    .await?
                    .take()
                    .0
            }
            pumicite_types::PipelineLayoutRef::Bindless => {
                let Some(heap) = self.heap.as_ref() else {
                    return Err(PipelineLoaderError::BindlessPluginNeededError);
                };
                heap.bindless_pipeline_layout().clone()
            }
        };

        let mut flags = vk::PipelineCreateFlags::empty();
        {
            if pipeline.disable_optimization {
                flags |= vk::PipelineCreateFlags::DISABLE_OPTIMIZATION;
            }
            if pipeline.dispatch_base {
                flags |= vk::PipelineCreateFlags::DISPATCH_BASE;
            }
        }

        let shader: LoadedAsset<ShaderModule> = load_context
            .loader()
            .immediate()
            .load(&pipeline.shader.path)
            .await?;
        let shader_flags = pipeline.shader.flags();
        let entry_point: CString = CString::new(pipeline.shader.entry_point)
            .map_err(|_| PipelineLoaderError::PipelineError("Invalid entry name"))?;

        let span = tracing::span!(
            tracing::Level::INFO,
            "Creating Compute Pipeline",
            path = load_context.asset_path().to_string()
        )
        .entered();
        let pipeline = self.pipeline_cache.create_compute_pipeline(
            layout,
            flags,
            &ShaderEntry {
                module: shader.get().0.clone(),
                entry: Cow::Owned(entry_point),
                flags: shader_flags,
                stage: vk::ShaderStageFlags::COMPUTE,
                specialization_info: Cow::Owned(Default::default()),
            },
        )?;
        span.exit();
        Ok(ComputePipeline(Arc::new(pipeline)))
    }

    fn extensions(&self) -> &[&str] {
        &[
            #[cfg(feature = "ron")]
            "comp.pipeline.ron",
            #[cfg(feature = "postcard")]
            "comp.pipeline.bin",
        ]
    }
}
