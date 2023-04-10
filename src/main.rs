fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = sapfire_renderer::Renderer::init_window(&event_loop);
    let renderer = sapfire_renderer::Renderer::new(&window);
    renderer.main_loop(window, event_loop);
}
