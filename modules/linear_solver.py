import pulp
import pandas as pd

def resolver_lp(df_obj, df_restr, tipo_opt):
    """
    Resuelve un problema de Programación Lineal Simple usando PuLP.
    """
    try:
        # 1. Definir el tipo de problema
        sentido = pulp.LpMaximize if tipo_opt == "Maximizar" else pulp.LpMinimize
        prob = pulp.LpProblem("Problema_IO_Moderno", sentido)

        # 2. Crear las Variables de Decisión (x1, x2, ..., xn)
        # Suponemos variables no negativas (>= 0) que es el estándar en IO1
        nombres_vars = list(df_obj.columns)
        variables = {v: pulp.LpVariable(v, lowBound=0, cat='Continuous') for v in nombres_vars}

        # 3. Definir la Función Objetivo
        # Suma producto de coeficientes * variables
        coefs_obj = df_obj.iloc[0].to_dict()
        prob += pulp.lpSum([coefs_obj[v] * variables[v] for v in nombres_vars]), "Funcion_Objetivo"

        # 4. Añadir Restricciones
        for i, row in df_restr.iterrows():
            # Expresión del lado izquierdo (LHS)
            lhs = pulp.lpSum([row[v] * variables[v] for v in nombres_vars])
            signo = row['Signo']
            rhs = row['RHS']

            if signo == "<=":
                prob += lhs <= rhs, f"Restriccion_{i}"
            elif signo == ">=":
                prob += lhs >= rhs, f"Restriccion_{i}"
            elif signo == "=":
                prob += lhs == rhs, f"Restriccion_{i}"

        # 5. Resolver el modelo
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # 6. Extraer Resultados con mayor seguridad
        status = pulp.LpStatus[prob.status]
        res_vars = {v: pulp.value(variables[v]) if pulp.value(variables[v]) is not None else 0.0 for v in nombres_vars}

        # Forzamos a que si el valor es None, devuelva 0.0 para evitar el error de formato
        valor_z = pulp.value(prob.objective)
        if valor_z is None:
            valor_z = 0.0

        # 7. Análisis de Sensibilidad (El toque Pro)
        # Precios Sombra (Shadow Prices / Dual Values)
        precios_sombra = []
        for name, c in prob.constraints.items():
            precios_sombra.append({"Restricción": name, "Precio Sombra": c.pi, "Slack": abs(c.slack)})

        return {
            "status": status,
            "z": valor_z,
            "variables": res_vars,
            "sensibilidad": pd.DataFrame(precios_sombra)
        }

    except Exception as e:
        return {"error": str(e)}