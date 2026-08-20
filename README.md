# hdb-resale-prediction

## Local Development
To run the server quickly, using Docker is highly recommended. See [Running the Docker Container](#running-the-docker-container) below.

To run it without Docker, follow the installation steps below. This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Please make sure you have uv installed.
```
uv sync

# Note: the entrypoint (for fastapi) for the server is defined in pyproject.toml
uv run fastapi dev
```


## Running the Docker container
In the root folder (same directory as `docker-compose.yml`), do

```
docker compose up -d --build
```

This will launch the docker containers for the `db` (postgres) and `backend`. After it starts up successfully, you can send any API requests via http://localhost:8000/.

For example: http://localhost:8000/inference/linear-regression-model?date=2026-08-13&floor_area_sqm=84&town=YISHUN&flat_type=4%20ROOM

To shut them down do `docker compose down`.
