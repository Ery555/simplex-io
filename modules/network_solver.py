import pulp
import pandas as pd

def limpiar_datos_red(df_redes):
    """Limpia filas vacías y estandariza nombres de nodos."""
    df = df_redes[(df_redes['Nodo Origen'] != "") & (df_redes['Nodo Destino'] != "")].copy()
    df['Nodo Origen'] = df['Nodo Origen'].astype(str).str.strip()
    df['Nodo Destino'] = df['Nodo Destino'].astype(str).str.strip()
    return df

def resolver_flujo_maximo(df_redes, origen, destino):
    """Resuelve el problema de Flujo Máximo con detalle de capacidades."""
    try:
        df = limpiar_datos_red(df_redes)
        nodos = list(set(df['Nodo Origen'].tolist() + df['Nodo Destino'].tolist()))
        
        if origen not in nodos or destino not in nodos:
            return {"error": f"El nodo '{origen}' o '{destino}' no existe en la red."}

        prob = pulp.LpProblem("Flujo_Maximo", pulp.LpMaximize)
        arcos = {(row['Nodo Origen'], row['Nodo Destino']): float(row['Capacidad']) for _, row in df.iterrows()}
        
        x = pulp.LpVariable.dicts("Ruta", arcos.keys(), lowBound=0, cat='Continuous')
        for (u, v) in arcos.keys():
            x[(u, v)].upBound = arcos[(u, v)]

        F = pulp.LpVariable("Flujo_Total", lowBound=0, cat='Continuous')
        prob += F, "Maximizar_el_Flujo"

        for n in nodos:
            entrada = pulp.lpSum([x[(u, v)] for (u, v) in arcos.keys() if v == n])
            salida = pulp.lpSum([x[(u, v)] for (u, v) in arcos.keys() if u == n])
            if n == origen:
                prob += (salida - entrada == F), f"Conservacion_Fuente_{n}"
            elif n == destino:
                prob += (entrada - salida == F), f"Conservacion_Sumidero_{n}"
            else:
                prob += (entrada - salida == 0), f"Conservacion_{n}"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]

        if status != "Optimal":
            return {"status": status}

        flujo_maximo = pulp.value(F)
        
        # Restaurando el detalle de rutas con capacidad sobrante
        lista_resultados = []
        for (u, v) in arcos.keys():
            flujo_usado = pulp.value(x[(u, v)])
            cap_max = arcos[(u, v)]
            holgura = cap_max - flujo_usado
            
            if flujo_usado > 0:
                lista_resultados.append({
                    "De": u,
                    "A": v,
                    "Flujo Enviado": flujo_usado,
                    "Capacidad Límite": cap_max,
                    "Capacidad Sobrante": holgura
                })

        df_resultados = pd.DataFrame(lista_resultados)
        if not df_resultados.empty:
            df_resultados.index = df_resultados.index + 1

        return {
            "status": status,
            "valor": flujo_maximo,
            "rutas": df_resultados
        }

    except Exception as e:
        return {"error": str(e)}

def resolver_ruta_mas_corta(df_redes, origen, destino):
    """Calcula la ruta con la menor suma de costos/distancias."""
    try:
        df = limpiar_datos_red(df_redes)
        nodos = list(set(df['Nodo Origen'].tolist() + df['Nodo Destino'].tolist()))
        
        if origen not in nodos or destino not in nodos:
            return {"error": f"El nodo '{origen}' o '{destino}' no existe en la red."}

        prob = pulp.LpProblem("Ruta_Mas_Corta", pulp.LpMinimize)
        arcos = {(row['Nodo Origen'], row['Nodo Destino']): float(row['Costo Unitario']) for _, row in df.iterrows()}
        x = pulp.LpVariable.dicts("ruta", arcos.keys(), cat='Binary')

        prob += pulp.lpSum([x[(u, v)] * arcos[(u, v)] for (u, v) in arcos.keys()])

        for n in nodos:
            entrada = pulp.lpSum([x[(u, v)] for (u, v) in arcos.keys() if v == n])
            salida = pulp.lpSum([x[(u, v)] for (u, v) in arcos.keys() if u == n])
            if n == origen:
                prob += salida - entrada == 1
            elif n == destino:
                prob += entrada - salida == 1
            else:
                prob += entrada - salida == 0

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        
        if status != "Optimal":
            return {"status": "Infactible"}

        res = [{"De": u, "A": v, "Distancia": arcos[(u, v)]} for (u, v) in arcos.keys() if pulp.value(x[(u, v)]) > 0.5]
        df_resultados = pd.DataFrame(res)
        if not df_resultados.empty:
            df_resultados.index = df_resultados.index + 1

        return {
            "status": "Optimal",
            "valor": pulp.value(prob.objective),
            "rutas": df_resultados
        }
    except Exception as e:
        return {"error": str(e)}