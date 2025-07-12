# Capstone/step11/search.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/search")
async def search(request: Request):
    try:
        data = await request.json()
        query_vec = data["query_vec"]
        k = data["k"]
        # Dummy implementation: return k zero vectors
        results = [[0.0] * len(query_vec) for _ in range(k)]
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        # Optionally log error for debugging
        return JSONResponse(content={"error": str(e)}, status_code=500)

# For local dev, you can run with: uvicorn Capstone.step11.search:app --reload
if __name__ == "__main__":
    uvicorn.run("Capstone.step11.search:app", host="0.0.0.0", port=8000, reload=True)