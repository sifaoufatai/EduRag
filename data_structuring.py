import os
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_all_structured_content(output_dir):
    """
    Charge tous les fichiers JSON dans un répertoire,
    extrait les séquences, contenus et métadonnées.

    :param output_dir: répertoire contenant les fichiers JSON
    :return: (contents, metadatas, all_sequences)
    """
    all_sequences = []

    for file in os.listdir(output_dir):
        if file.endswith(".json"):  # sécurité
            with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                content = json.load(f)
                all_sequences.extend(content)

    metadatas = [
        {
            "Situation d’Apprentissage": seq.get("sa", ""),
            "sequence": seq.get("sequence", ""),
            "grade": seq.get("grade", "")
        }
        for seq in all_sequences
    ]

    sequences_content = [seq.get("content", "") for seq in all_sequences]

    return sequences_content, metadatas, all_sequences


def store_in_chroma(output_dir, persist_dir="chroma_db"):
    """
    Extrait les séquences et les stocke dans une base Chroma locale.
    """
    # Récupérer contenus et métadonnées
    sequences_content, metadatas, _ = create_all_structured_content(output_dir)

    # Embeddings HuggingFace (à jour)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Création de la DB Chroma (persistance auto depuis 0.4.x)
    vectordb = Chroma.from_texts(
        texts=sequences_content,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir
    )

    return vectordb


if __name__ == "__main__":
    output_dir = "output"
    sequences_content, metadatas, all_sequences = create_all_structured_content(output_dir)

    with open("all_sequences.json", "w", encoding="utf-8") as f:
        json.dump(all_sequences, f, ensure_ascii=False, indent=4)
    with open("all_sequences_content.json", "w", encoding="utf-8") as f:
        json.dump(sequences_content, f, ensure_ascii=False, indent=4)
    with open("all_sequences_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=4)

    print("Nombre de séquences :", len(all_sequences))
    print("Nombre de contenus :", len(sequences_content))
    print("Nombre de métadonnées :", len(metadatas))

    vectordb = store_in_chroma(output_dir)
