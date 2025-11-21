# 🌱 Cells - Cellules KIBALI

"""Module des cellules KIBALI

Contient les définitions de cellules en langage KIBALI :
- Cellules biologiques autonomes
- Cellules d'entraînement RAG
- Cellules spécialisées
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import re

class ChargeurCellules:
    """Chargeur de cellules KIBALI depuis les fichiers .kib"""

    def __init__(self, chemin_cells: str = "lib/src/cells"):
        """Initialise le chargeur de cellules

        Args:
            chemin_cells: Chemin vers le répertoire des cellules
        """
        self.chemin_cells = Path(chemin_cells)
        self.cellules_chargees: Dict[str, Dict[str, Any]] = {}

    def charger_cellules(self) -> Dict[str, Dict[str, Any]]:
        """Charge toutes les cellules depuis les fichiers .kib

        Returns:
            Dict: Cellules chargées avec leurs configurations
        """
        if not self.chemin_cells.exists():
            return {}

        cellules = {}
        for fichier_kib in self.chemin_cells.glob("*.kib"):
            try:
                cellule = self._parser_cellule(fichier_kib)
                if cellule:
                    cellules[cellule["nom"]] = cellule
            except Exception as e:
                print(f"Erreur chargement {fichier_kib}: {e}")

        self.cellules_chargees = cellules
        return cellules

    def charger_cellule(self, nom_cellule: str) -> Optional[Dict[str, Any]]:
        """Charge une cellule spécifique

        Args:
            nom_cellule: Nom de la cellule à charger

        Returns:
            Dict: Configuration de la cellule ou None si non trouvée
        """
        fichier_kib = self.chemin_cells / f"{nom_cellule}.kib"
        if not fichier_kib.exists():
            return None

        return self._parser_cellule(fichier_kib)

    def _parser_cellule(self, fichier: Path) -> Optional[Dict[str, Any]]:
        """Parse un fichier cellule .kib

        Args:
            fichier: Chemin vers le fichier .kib

        Returns:
            Dict: Configuration parsée de la cellule
        """
        contenu = fichier.read_text(encoding='utf-8')

        # Extraction du nom de la cellule
        match_nom = re.search(r'cellule\s+(\w+)', contenu)
        if not match_nom:
            return None

        nom_cellule = match_nom.group(1)

        # Extraction des propriétés
        proprietes = {}
        for ligne in contenu.split('\n'):
            ligne = ligne.strip()
            if ':' in ligne and not ligne.startswith('//'):
                cle, valeur = ligne.split(':', 1)
                cle = cle.strip()
                valeur = valeur.strip()

                # Conversion de type basique
                if valeur.isdigit():
                    proprietes[cle] = int(valeur)
                elif valeur.replace('.', '').isdigit():
                    proprietes[cle] = float(valeur)
                elif valeur.lower() in ['true', 'false']:
                    proprietes[cle] = valeur.lower() == 'true'
                else:
                    proprietes[cle] = valeur

        # Extraction des actions
        actions = []
        for ligne in contenu.split('\n'):
            ligne = ligne.strip()
            if ligne.startswith('action ') and '(' in ligne:
                action_match = re.search(r'action\s+(\w+)\s*\(', ligne)
                if action_match:
                    actions.append(action_match.group(1))

        return {
            "nom": nom_cellule,
            "fichier": str(fichier),
            "proprietes": proprietes,
            "actions": actions,
            "contenu_brut": contenu
        }

    def lister_cellules(self) -> List[str]:
        """Liste toutes les cellules disponibles

        Returns:
            List: Noms des cellules disponibles
        """
        if not self.chemin_cells.exists():
            return []

        return [f.stem for f in self.chemin_cells.glob("*.kib")]

# Instance globale
_chargeur_global = None

def charger_cellule(nom_cellule: str) -> Optional[Dict[str, Any]]:
    """Fonction utilitaire pour charger une cellule

    Args:
        nom_cellule: Nom de la cellule

    Returns:
        Dict: Configuration de la cellule
    """
    global _chargeur_global
    if _chargeur_global is None:
        _chargeur_global = ChargeurCellules()

    return _chargeur_global.charger_cellule(nom_cellule)

def lister_cellules() -> List[str]:
    """Fonction utilitaire pour lister les cellules

    Returns:
        List: Noms des cellules
    """
    global _chargeur_global
    if _chargeur_global is None:
        _chargeur_global = ChargeurCellules()

    return _chargeur_global.lister_cellules()