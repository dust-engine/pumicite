.PHONY: default
default: mesh_shading.spv

mesh_shading.spv: mesh_shading.slang
	slangc -O3 -target spirv -o mesh_shading.spv mesh_shading.slang
