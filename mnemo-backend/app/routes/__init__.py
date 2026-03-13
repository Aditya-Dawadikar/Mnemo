from routes.chat_routes import router as chat_router
from routes.health_routes import router as health_router
from routes.memory_routes import router as memory_router

__all__ = ["chat_router", "health_router", "memory_router"]