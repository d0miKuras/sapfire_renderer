fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let _window = sapfire_renderer::Renderer::init_window(&event_loop);
    sapfire_renderer::Renderer::main_loop(event_loop);
}
