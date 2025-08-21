import os
import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path=None):
        # Default CSV location (relative to this file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = file_path or os.path.join(base_dir, "resource", "my_portfolio.csv")

        # Load CSV
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Portfolio CSV not found at {self.file_path}")

        self.data = pd.read_csv(self.file_path)

        # Ensure required columns exist
        required_cols = {"Techstack", "Links"}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(
                f"CSV must contain columns: {required_cols}, but found {list(self.data.columns)}"
            )

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path="vectorstore")
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Load portfolio data into ChromaDB (only if collection is empty)."""
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                techstack = row["Techstack"]

                # Ensure techstack is always a clean string
                if isinstance(techstack, list):
                    techstack_str = ", ".join(map(str, techstack))
                else:
                    techstack_str = str(techstack)

                # Add to ChromaDB
                self.collection.add(
                    documents=[techstack_str],
                    metadatas={"links": str(row["Links"])},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        """Query ChromaDB for matching skills and return associated links."""
        if not isinstance(skills, str):
            skills = str(skills)

        results = self.collection.query(query_texts=[skills], n_results=2)
        return results.get("metadatas", [])
