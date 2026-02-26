.PHONY: default
default: mesh_shader_culling.spv

triangle.spv: mesh_shader_culling.slang
	slangc -O3 -target spirv -o mesh_shader_culling.spv mesh_shader_culling.slang
