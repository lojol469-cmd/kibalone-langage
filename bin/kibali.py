# bin/kibali.py
import sys
import os
sys.path.append(os.path.dirname(__file__) + "/..")

from lib.kibali import KibaliRuntime
import json
import time

def animate(prompt):
    print(f"\nKibalone Animation Engine démarre...")
    print(f"Prompt: {prompt}\n")

    rt = KibaliRuntime()
    rt.load_brain()  # Phi-1.5 + CodeLlama auto-détectés

    # Charger les cellules 3D de base
    print("Chargement des nano-IA 3D...")
    for cell_file in os.listdir("cells/3d"):
        if cell_file.endswith(".kib"):
            rt.load_cell(f"cells/3d/{cell_file}")
            print(f"  → {cell_file} chargée")

        # Utiliser le Directeur IA (CodeLlama) pour générer le scénario
        print("\nLe Réalisateur IA réfléchit avec CodeLlama-7B...")
        scenario_prompt = f"""
Tu es un réalisateur de film d'animation style Studio Ghibli.
Crée un court scénario de 30 secondes basé sur cela :
"{prompt}"

Réponds UNIQUEMENT en JSON valide avec cette structure :
{{
  "titre": "string",
  "personnages": ["Nom1", "Nom2"],
  "actions": [
    {{"temps": 0, "action": "Camera zoom sur le personnage"}},
    {{"temps": 5, "action": "Le personnage marche dans la forêt"}},
    {{"temps": 10, "action": "Les arbres bougent avec le vent"}},
    {{"temps": 20, "action": "Plan large magique"}}
  ],
  "style": "organique, poétique, vivant"
}}

Les cellules nano-AI ont maintenant accès aux outils puissants :
- draw_image() : génération d'images SDXL
- generate_voice() : synthèse vocale XTTS-v2
- create_3d_model() : modèles 3D TripoSR
- render_frame() : rendu Three.js
- write_kib_cell() : création de nouvelles cellules
"""
    try:
        response = rt.query_brain(
            scenario_prompt,
            brain="codellama-7b",
            max_tokens=1024,
            temperature=0.9
        )
        # Nettoyer la réponse
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        scenario_json = json.loads(response[json_start:json_end])
        print("Scénario généré avec succès !")
        print(json.dumps(scenario_json, indent=2, ensure_ascii=False))

        print(f"\n🎬 Animation en cours de création...")
        time.sleep(2)
        print(f"Sortie : outputs/{scenario_json['titre'].replace(' ', '_')}.mp4")
        print("Animation terminée ! (MVP Phase 1)")

        # PHASE 2: Lancer le visualiseur 3D temps réel
        print("\n🚀 PHASE 2: Démarrage du rendu 3D temps réel...")
        launch_3d_viewer(scenario_json)

    except Exception as e:
        print("CodeLlama n'a pas répondu en JSON → fallback simple")
        print(f"""
Scénario généré (mode simple) :
Titre: Exploration Mystérieuse
Personnage marche dans une forêt vivante
Arbres qui dansent avec le vent
Ambiance magique et organique
        """)

        # Fallback avec visualiseur 3D quand même
        fallback_scenario = {
            "titre": "Forêt Mystérieuse",
            "personnages": ["Explorateur"],
            "actions": [
                {"temps": 0, "action": "Camera zoom"},
                {"temps": 5, "action": "Marche dans la forêt"},
                {"temps": 10, "action": "Arbres bougent"}
            ],
            "style": "organique, vivant"
        }
        launch_3d_viewer(fallback_scenario)

def launch_3d_viewer(scenario):
    """Lance le visualiseur 3D avec le scénario"""
    print(f"\n🎭 Lancement du visualiseur 3D pour: {scenario['titre']}")

    try:
        # Importer et lancer le serveur 3D
        import subprocess
        import sys

        # Lancer le serveur 3D en arrière-plan
        cmd = [sys.executable, "bin/3d_server.py", json.dumps(scenario)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("✅ Serveur 3D lancé!")
        print("🌐 Le visualiseur va s'ouvrir dans votre navigateur")
        print("🎮 Contrôles: La caméra suit automatiquement l'action")
        print("🎨 Style: Animation organique avec IA")

        # Attendre un peu pour que le serveur démarre
        time.sleep(3)

        return True

    except Exception as e:
        print(f"❌ Erreur lancement 3D: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "animate":
        print("Usage: python bin/kibali.py animate \"votre description\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[2:])[1:-1]  # enlever les guillemets
    animate(prompt)