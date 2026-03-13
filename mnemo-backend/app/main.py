from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.chat_routes import router as chat_router
from routes.health_routes import router as health_router
from routes.memory_routes import router as memory_router


def create_app() -> FastAPI:
	app = FastAPI(title="Mnemo Backend API")

	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	app.include_router(health_router)
	app.include_router(chat_router)
	app.include_router(memory_router)

	return app


app = create_app()
