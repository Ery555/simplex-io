#FUNCION GENERADORA DE REPORTES MODULO PROGRAMACION LINEAL SIMPLEX

def generar_reporte_texto(tipo_mod, opt, z, vars_dict, sensibil_df=None):
    """Crea un reporte de texto con formato académico."""
    reporte = f"""
============================================================
       UNIVERSIDAD MAYOR DE SAN ANDRÉS (UMSA)
      FACULTAD DE CIENCIAS PURAS Y NATURALES
           SUITE DE INVESTIGACIÓN OPERATIVA
============================================================

REPORTE DE RESULTADOS - {tipo_mod.upper()}
------------------------------------------------------------
TIPO DE OPTIMIZACIÓN: {opt}
VALOR ÓPTIMO (Z): {z:,.2f}

DETALLE DE VARIABLES DE DECISIÓN:
------------------------------------------------------------
"""
    for var, val in vars_dict.items():
        reporte += f"{var.ljust(15)}: {val:>10.2f}\n"

    if sensibil_df is not None:
        reporte += "\nANÁLISIS DE SENSIBILIDAD / DUALIDAD:\n"
        reporte += "------------------------------------------------------------\n"
        reporte += sensibil_df.to_string(index=False)
    
    reporte += "\n\n============================================================"
    
    return reporte