from supabase import create_client, Client
import os

def get_supabase_client() -> Client:
    """Create and return a Supabase client using environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    return create_client(url, key)