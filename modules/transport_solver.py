import pulp
import pandas as pd

def resolver_transporte(df_transp, tipo_opt):
    """
    Resuelve el Modelo de Transporte. 
    Aplica balanceo automático con nodos ficticios si es necesario.
    """
    try:
        # 1. Separar la matriz en listas y diccionarios
        origenes = [idx for idx in df_transp.index if idx != 'Demanda']
        destinos = [col for col in df_transp.columns if col != 'Oferta']

        oferta = {o: df_transp.at[o, 'Oferta'] for o in origenes}
        demanda = {d: df_transp.at['Demanda', d] for d in destinos}

        # Matriz de costos: costos[origen][destino]
        costos = {o: {d: df_transp.at[o, d] for d in destinos} for o in origenes}

        # 2. LÓGICA DE IO1: BALANCEO AUTOMÁTICO
        total_oferta = sum(oferta.values())
        total_demanda = sum(demanda.values())
        es_balanceado = (total_oferta == total_demanda)

        if total_oferta > total_demanda:
            # Creamos un Destino Ficticio para absorber el exceso de Oferta
            dest_ficticio = "Dest_Ficticio"
            destinos.append(dest_ficticio)
            demanda[dest_ficticio] = total_oferta - total_demanda
            for o in origenes:
                costos[o][dest_ficticio] = 0.0  # Costo 0 hacia el nodo ficticio

        elif total_demanda > total_oferta:
            # Creamos un Origen Ficticio para suplir el exceso de Demanda
            orig_ficticio = "Orig_Ficticio"
            origenes.append(orig_ficticio)
            oferta[orig_ficticio] = total_demanda - total_oferta
            costos[orig_ficticio] = {d: 0.0 for d in destinos}

        # 3. Construcción del Modelo en PuLP
        sentido = pulp.LpMaximize if tipo_opt == "Maximizar" else pulp.LpMinimize
        prob = pulp.LpProblem("Modelo_Transporte", sentido)

        # Variables de Decisión x_ij (Matriz de envíos)
        rutas = [(o, d) for o in origenes for d in destinos]
        x = pulp.LpVariable.dicts("Ruta", (origenes, destinos), lowBound=0, cat='Continuous')

        # 4. Función Objetivo: Sumatoria de (Costo_ij * x_ij)
        prob += pulp.lpSum([x[o][d] * costos[o][d] for (o, d) in rutas]), "Costo_Total"

        # 5. Restricciones
        # Restricciones de Oferta (Lo que sale == Lo que tengo)
        for o in origenes:
            prob += pulp.lpSum([x[o][d] for d in destinos]) == oferta[o], f"Oferta_{o}"

        # Restricciones de Demanda (Lo que llega == Lo que necesito)
        for d in destinos:
            prob += pulp.lpSum([x[o][d] for o in origenes]) == demanda[d], f"Demanda_{d}"

        # 6. Resolver el modelo
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]

        # 7. Extraer Resultados
        valor_z = pulp.value(prob.objective)
        if valor_z is None: valor_z = 0.0

        # Formatear las variables como un DataFrame (Matriz) para que se vea igual que la entrada
        res_data = []
        for o in origenes:
            fila = []
            for d in destinos:
                val = pulp.value(x[o][d])
                fila.append(val if val is not None else 0.0)
            res_data.append(fila)

        df_resultados = pd.DataFrame(res_data, index=origenes, columns=destinos)

        # Extraer Duales (En transporte se llaman Multiplicadores u_i y v_j)
        duales_oferta = {name.replace("Oferta_", ""): c.pi for name, c in prob.constraints.items() if "Oferta" in name}
        duales_demanda = {name.replace("Demanda_", ""): c.pi for name, c in prob.constraints.items() if "Demanda" in name}

        return {
            "status": status,
            "z": valor_z,
            "matriz_resultados": df_resultados,
            "balanceado": es_balanceado,
            "duales_oferta": duales_oferta,
            "duales_demanda": duales_demanda
        }

    except Exception as e:
        return {"error": str(e)}