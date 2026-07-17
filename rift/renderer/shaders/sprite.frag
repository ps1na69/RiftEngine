#version 460 core
#extension GL_ARB_bindless_texture : require

layout(location = 0) in vec2 in_uv;
layout(location = 1) in flat uvec2 in_tex_handle;

layout(location = 0) out vec4 out_color;

void main() {
    sampler2D tex = sampler2D(in_tex_handle);
    vec4 color = texture(tex, in_uv);
    if (color.a < 0.01) {
        discard;
    }
    out_color = color;
}
