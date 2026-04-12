import pulp
import pandas as pd

def resolver_asignacion(df_asign, tipo_opt):
    """
    Resuelve el Modelo de Asignación.
    Aplica balanceo automático si Trabajadores != Tareas.
    """
    try:
        trabajadores_originales = list(df_asign.index)
        tareas_originales = list(df_asign.columns)
        
        n_w = len(trabajadores_originales)
        n_t = len(tareas_originales)
        es_balanceado = (n_w == n_t)

        # 1. Balanceo Automático (Cuadrar la matriz)
        trabajadores = trabajadores_originales.copy()
        tareas = tareas_originales.copy()
        
        # Matriz de costos inicial
        costos = {w: {t: df_asign.at[w, t] for t in tareas_originales} for w in trabajadores_originales}

        if n_w > n_t:
            # Más trabajadores que tareas: Agregar Tareas Ficticias
            diferencia = n_w - n_t
            for i in range(diferencia):
                tarea_ficticia = f"Tarea_Ficticia_{i+1}"
                tareas.append(tarea_ficticia)
                for w in trabajadores:
                    costos[w][tarea_ficticia] = 0.0 # Costo 0
                    
        elif n_t > n_w:
            # Más tareas que trabajadores: Agregar Trabajadores Ficticios
            diferencia = n_t - n_w
            for i in range(diferencia):
                trab_ficticio = f"Trab_Ficticio_{i+1}"
                trabajadores.append(trab_ficticio)
                costos[trab_ficticio] = {t: 0.0 for t in tareas}

        # 2. Construcción del Modelo en PuLP
        sentido = pulp.LpMaximize if tipo_opt == "Maximizar" else pulp.LpMinimize
        prob = pulp.LpProblem("Modelo_Asignacion", sentido)

        # Variables: Binarias (0 o 1) porque es asignación 1 a 1
        rutas = [(w, t) for w in trabajadores for t in tareas]
        x = pulp.LpVariable.dicts("Asign", (trabajadores, tareas), cat='Binary')

        # Función Objetivo
        prob += pulp.lpSum([costos[w][t] * x[w][t] for (w, t) in rutas]), "Z"

        # Restricciones
        # 1. Cada trabajador hace exactamente 1 tarea
        for w in trabajadores:
            prob += pulp.lpSum([x[w][t] for t in tareas]) == 1, f"Trabajador_{w}"
            
        # 2. Cada tarea es hecha por exactamente 1 trabajador
        for t in tareas:
            prob += pulp.lpSum([x[w][t] for w in trabajadores]) == 1, f"Tarea_{t}"

        # 3. Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        
        valor_z = pulp.value(prob.objective)
        if valor_z is None: valor_z = 0.0

        # 4. Formatear Resultados
        lista_asignaciones = []
        res_matriz = []

        for w in trabajadores:
            fila_matriz = []
            for t in tareas:
                # pulp.value(x[w][t]) debería ser 1.0 o 0.0
                val = pulp.value(x[w][t])
                val_limpio = 1.0 if val is not None and val > 0.5 else 0.0
                fila_matriz.append(val_limpio)
                
                # Si está asignado, lo añadimos a la lista amigable
                if val_limpio == 1.0:
                    costo_real = costos[w][t]
                    # Solo mostramos en la lista si no es ficticio (o podemos mostrar todo para transparencia)
                    lista_asignaciones.append({
                        "Trabajador / Agente": w.replace("_", " "),
                        "Tarea Asignada": t.replace("_", " "),
                        "Costo / Aporte": costo_real
                    })
            res_matriz.append(fila_matriz)

        df_matriz = pd.DataFrame(res_matriz, index=trabajadores, columns=tareas)
        df_lista = pd.DataFrame(lista_asignaciones)
        
        # Ajustar índices para que empiecen en 1 (Estética académica)
        df_lista.index = df_lista.index + 1

        return {
            "status": status,
            "z": valor_z,
            "balanceado": es_balanceado,
            "n_trabajadores": n_w,
            "n_tareas": n_t,
            "lista_asignaciones": df_lista,
            "matriz_resultados": df_matriz
        }

    except Exception as e:
        return {"error": str(e)}