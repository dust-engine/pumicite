.PHONY: default
default: pbr.spv

pbr.spv: pbr.slang
	slangc -g -O3 -target spirv -o pbr.spv pbr.slang
