from markdown_it import MarkdownIt
import json
import os as os
import re

def parses_md_to_json(fiel_dir, output_dir="output"):
    for file in os.listdir(fiel_dir):
        if file.endswith(".md"):
            file_base_name =os.path.basename(file)
            file_base_name = os.path.splitext(file_base_name)[0]
            json_file_name = file_base_name + ".json"
            with open(os.path.join(fiel_dir, file),  "r") as f:
                md = MarkdownIt()
                tokens = md.parse(f.read())
                print(json.dumps([t.as_dict() for t in tokens], indent=2, ensure_ascii=False))
                with open(os.path.join(output_dir, json_file_name), "w") as j:
                    j.write(json.dumps([t.as_dict() for t in tokens], indent=2, ensure_ascii=False))



def extract_text(file_dir, output_dir):
    """
    Extract sequences and situations d'apprentissage (SA) from markdown files.

    Assume each sequence has:
        - sequence title
        - sequence description
        - sequence content
        - list of properties
        - SA (situation d’apprentissage) defined within it

    :param file_dir: directory containing .md files
    :param output_dir: directory to save .json output
    :return: list of all sequences parsed
    """
    pattern_sequence = "#### **Sequence"
    pattern_sa = "### **Situation d’Apprentissage"

    patterns_sequence = re.compile(r"^#+\s*\*\*?Sequence", re.IGNORECASE)
    patterns_sa = re.compile(r"^#+\s*\*\*?Situation d[’']Apprentissage",
                            re.IGNORECASE)

    all_sequence = []

    for file in os.listdir(file_dir):
        if file.endswith(".md"):
            file_base_name = os.path.splitext(file)[0]
            json_file_name = file_base_name + ".json"

            current_sequence = ""
            current_sa = ""
            current_content = ""

            with open(os.path.join(file_dir, file), "r", encoding="utf-8") as f:
                lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith(pattern_sa) or re.match(patterns_sa, line):
                        if current_content:
                            all_sequence.append({
                                "sequence": current_sequence,
                                "sa": current_sa,
                                "content": current_content.strip(),
                                "grade": file_base_name
                            })
                            current_content = ""
                            current_sequence = ""
                        current_sa = line
                        continue

                    if line.startswith(pattern_sequence) or re.match(patterns_sequence, line):
                        if current_content:
                            all_sequence.append({
                                "sequence": current_sequence,
                                "sa": current_sa,
                                "content": current_content.strip(),
                                "grade": file_base_name
                            })
                            current_content = ""

                        current_sequence = line
                        continue

                    if current_sequence:
                        current_content += line + "\n"

                if current_content:
                    all_sequence.append({
                        "sequence": current_sequence,
                        "sa": current_sa,
                        "content": current_content.strip(),
                        "grade": file_base_name
                    })

            with open(os.path.join(output_dir, json_file_name), "w", encoding="utf-8") as jf:
                json.dump(all_sequence, jf, ensure_ascii=False, indent=2)

    return all_sequence











if __name__ == "__main__":
    #parses_md_to_json("/home/INT/idrissou.f/PycharmProjects/EduRAG/EDUMDFILE")
    extract_text("/home/INT/idrissou.f/PycharmProjects/EduRAG/EDUMDFILE", "/home/INT/idrissou.f/PycharmProjects/EduRAG/output")

