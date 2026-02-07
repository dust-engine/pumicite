use bevy_ecs::{
    entity::MapEntities,
    prelude::*,
    reflect::{ReflectComponent, ReflectMapEntities},
};
use bevy_reflect::Reflect;
use bevy_pumicite::pumicite::buffer::Buffer;
use bevy_pumicite::pumicite::{ash::vk, buffer::BufferLike};
use std::sync::Arc;
use std::{collections::BTreeMap, ops::Deref};

pub mod gltf;

#[derive(Component, Default, Clone, Reflect, MapEntities)]
#[reflect(opaque, Component, MapEntities)]
pub struct Model {
    #[entities]
    primitives: Vec<Primitive>,
}
impl Deref for Model {
    type Target = [Primitive];

    fn deref(&self) -> &Self::Target {
        &self.primitives
    }
}

/// List of all instances for this model
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
#[relationship_target(relationship = InstanceOf)]
pub struct ModelInstances(Vec<Entity>);
impl ModelInstances {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Entity> {
        self.0.iter().cloned()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// Types implementing this trait should uniquely map to a singular u64 key.
pub trait PrimitiveKey {
    fn from_u64(key: u64) -> Self;
    fn to_u64(&self) -> u64;
}
#[derive(Clone, MapEntities, Reflect)]
#[reflect(opaque, MapEntities)]
pub struct Primitive {
    pub topology: vk::PrimitiveTopology,
    pub index_count: u32,
    pub index_type: vk::IndexType,

    pub index_buffer: Option<Arc<Buffer>>,
    pub index_buffer_offset: u64,

    attributes: BTreeMap<u64, Attribute>,

    #[entities]
    pub material: Entity,
}
impl Default for Primitive {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::POINT_LIST,
            index_count: 0,
            index_type: vk::IndexType::UINT16,
            index_buffer: None,
            index_buffer_offset: 0,
            attributes: BTreeMap::new(),
            material: Entity::PLACEHOLDER,
        }
    }
}

impl Primitive {
    pub fn attribute(&self, key: impl PrimitiveKey) -> Option<&Attribute> {
        self.attributes.get(&key.to_u64())
    }
    /// Returns the GPUVA of the attribute buffer, or 0 if the provided key doesn't exist.
    pub fn attribute_gpuva(&self, key: impl PrimitiveKey) -> u64 {
        self.attribute(key)
            .map(|x| x.buffer.device_address() + x.offset as u64)
            .unwrap_or_default()
    }
}
#[derive(Reflect, Clone, Debug)]
#[reflect(opaque)]
pub struct Attribute {
    pub buffer: Arc<Buffer>,
    pub offset: usize,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
#[relationship(relationship_target = ModelInstances)]
pub struct InstanceOf {
    #[relationship]
    pub model: Entity,
}

#[derive(Component, Reflect, Clone, MapEntities)]
#[reflect(Component, MapEntities)]
#[reflect(opaque)]
pub struct Scene {
    #[entities]
    models: Vec<Entity>,
    #[entities]
    instances: Vec<Entity>,
}
impl Scene {
    pub fn models(&self) -> &[Entity] {
        &self.models
    }
    pub fn instances(&self) -> &[Entity] {
        &self.instances
    }
}
