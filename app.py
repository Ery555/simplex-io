import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import base64
import os
from streamlit_option_menu import option_menu
from modules.linear_solver import resolver_lp
from modules.generador_reportes import generar_reporte_pdf
from modules.generador_reportes import generar_reporte_pdf_transporte
from modules.generador_reportes import generar_reporte_pdf_asignacion
from modules.generador_reportes import generar_reporte_pdf_redes


# Configuración de la página
st.set_page_config(page_title="IO Modern Solver - UMSA", layout="wide")
st.markdown("""
    <style>
    /* Estilizar las métricas para que parezcan tarjetas modernas */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 5% 10%;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    /* Ocultar el botón de menú de Streamlit por defecto para un look más "App" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# Inyectar CDN de Bootstrap Icons para usar en toda la página
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

def titulo_con_icono(texto, icono, nivel="h2"):
    """
    Genera un encabezado HTML con un ícono de Bootstrap.
    Nivel h2 equivale a st.header, h3 equivale a st.subheader.
    """
    html = f"""
    <{nivel} style='color: #F4F4F5; font-weight: 600; margin-bottom: 1rem;'>
        <i class='bi bi-{icono}' style='color: #00D287; margin-right: 10px;'></i>{texto}
    </{nivel}>
    """
    st.markdown(html, unsafe_allow_html=True)

def obtener_base64_de_archivo(ruta_archivo):
    with open(ruta_archivo, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
# --- 1. INICIALIZACIÓN GLOBAL PROG LINEAL ---
if 'df_obj' not in st.session_state:
    st.session_state.df_obj = pd.DataFrame([[0.0, 0.0]], columns=['x1', 'x2'], index=['Coeficiente'])
if 'df_restr' not in st.session_state:
    st.session_state.df_restr = pd.DataFrame(
        {'x1': [0.0, 0.0], 'x2': [0.0, 0.0], 'Signo': ['<=', '<='], 'RHS': [0.0, 0.0]},
        index=['R1', 'R2']
    )
if 'tipo_opt' not in st.session_state:
    st.session_state.tipo_opt = "Maximizar"
# --- INICIALIZACIÓN GLOBAL TRANSPORTE ---
if 'df_transp' not in st.session_state:
    # Matriz inicial 2x2. Incluye Oferta y Demanda.
    # La esquina inferior derecha [Demanda, Oferta] se queda en 0 (no se usa)
    data_t = {
        'D1': [0.0, 0.0, 0.0],
        'D2': [0.0, 0.0, 0.0],
        'Oferta': [0.0, 0.0, 0.0] 
    }
    st.session_state.df_transp = pd.DataFrame(data_t, index=['O1', 'O2', 'Demanda'])

if 'tipo_opt_transp' not in st.session_state:
    st.session_state.tipo_opt_transp = "Minimizar"

# --- INICIALIZACIÓN: ASIGNACIÓN ---
if 'df_asign' not in st.session_state:
    # Matriz inicial limpia 3x3 (Trabajadores vs Tareas)
    st.session_state.df_asign = pd.DataFrame(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=['Trabajador 1', 'Trabajador 2', 'Trabajador 3'],
        columns=['Tarea 1', 'Tarea 2', 'Tarea 3']
    )
if 'tipo_opt_asign' not in st.session_state:
    st.session_state.tipo_opt_asign = "Minimizar"


# --- INICIALIZACIÓN: FLUJO DE REDES ---
if 'df_redes' not in st.session_state:
    # Columnas estándar para grafos: Desde, Hacia, Capacidad, Costo
    st.session_state.df_redes = pd.DataFrame(
        [["O", "A", 10.0, 2.0], ["O", "B", 15.0, 5.0], ["A", "T", 10.0, 1.0]],
        columns=['Nodo Origen', 'Nodo Destino', 'Capacidad', 'Costo Unitario']
    )
if 'tipo_problema_redes' not in st.session_state:
    st.session_state.tipo_problema_redes = "Flujo Máximo"

# FUNCIONES DE AJUSTE DE DIMENSIONES

# PROG LINEAL
def ajustar_dimensiones():
    """
    Sincroniza los cambios del diccionario del editor con el DataFrame 
    antes de redimensionar para evitar el AttributeError y el parpadeo.
    """
    # A. Sincronizar Función Objetivo
    if 'editor_obj' in st.session_state:
        # Extraemos solo las filas editadas del diccionario
        edits = st.session_state.editor_obj.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_obj.at[st.session_state.df_obj.index[row_idx], col] = val

    # B. Sincronizar Restricciones
    if 'editor_restr' in st.session_state:
        edits_r = st.session_state.editor_restr.get('edited_rows', {})
        for row_idx, changes in edits_r.items():
            for col, val in changes.items():
                st.session_state.df_restr.at[st.session_state.df_restr.index[row_idx], col] = val

    # C. Redimensionamiento Seguro
    n_v = st.session_state.n_vars_slider
    n_c = st.session_state.n_cons_slider
    new_vars = [f"x{i+1}" for i in range(n_v)]
    new_restr = [f"R{i+1}" for i in range(n_c)]

    # Reindexar usando el DataFrame real (ya actualizado con los cambios)
    st.session_state.df_obj = st.session_state.df_obj.reindex(columns=new_vars, fill_value=0.0)
    
    cols_restr = new_vars + ["Signo", "RHS"]
    st.session_state.df_restr = st.session_state.df_restr.reindex(columns=cols_restr, index=new_restr)
    
    # Rellenar valores por defecto en nuevas celdas
    st.session_state.df_restr[new_vars] = st.session_state.df_restr[new_vars].fillna(0.0)
    st.session_state.df_restr["Signo"] = st.session_state.df_restr["Signo"].fillna("<=")
    st.session_state.df_restr["RHS"] = st.session_state.df_restr["RHS"].fillna(0.0)

# --- TRANSPORTE ---
def ajustar_dimensiones_transp():
    """Redimensiona la matriz de Transporte protegiendo Oferta y Demanda"""
    # 1. Sincronizar datos manuales
    if 'editor_transp' in st.session_state:
        edits = st.session_state.editor_transp.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_transp.at[st.session_state.df_transp.index[row_idx], col] = val

    n_o = st.session_state.n_orig_slider
    n_d = st.session_state.n_dest_slider
    
    df = st.session_state.df_transp.copy()

    # 2. "Despegar" la fila de Demanda y columna de Oferta
    demanda_row = df.loc[['Demanda']].copy()
    df = df.drop(index='Demanda')
    
    oferta_col = df[['Oferta']].copy()
    df = df.drop(columns='Oferta')
    
    # 3. Redimensionar la matriz central de Costos
    new_origs = [f"O{i+1}" for i in range(n_o)]
    new_dests = [f"D{i+1}" for i in range(n_d)]
    df = df.reindex(index=new_origs, columns=new_dests, fill_value=0.0)
    
    # 4. Redimensionar los bordes
    demanda_row = demanda_row.drop(columns='Oferta').reindex(columns=new_dests, fill_value=0.0)
    oferta_col = oferta_col.reindex(index=new_origs, fill_value=0.0)
    
    # 5. Volver a "Pegar" todo
    df['Oferta'] = oferta_col['Oferta']
    demanda_row['Oferta'] = 0.0 # Esquina vacía
    st.session_state.df_transp = pd.concat([df, demanda_row])

# ASIGNACION

def ajustar_dimensiones_asign():
    """Redimensiona la matriz de Asignación guardando los datos previos."""
    # 1. Sincronizar edición manual
    if 'editor_asign' in st.session_state:
        edits = st.session_state.editor_asign.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_asign.at[st.session_state.df_asign.index[row_idx], col] = val

    # 2. Redimensionar
    n_w = st.session_state.n_workers_slider
    n_t = st.session_state.n_tasks_slider
    
    new_workers = [f"Trabajador {i+1}" for i in range(n_w)]
    new_tasks = [f"Tarea {i+1}" for i in range(n_t)]
    
    st.session_state.df_asign = st.session_state.df_asign.reindex(
        index=new_workers, columns=new_tasks, fill_value=0.0
    )

# FLUJO DE REDES

def ajustar_dimensiones_redes():
    """Redimensiona la tabla de arcos (filas) para el modelo de Redes."""
    if 'editor_redes' in st.session_state:
        edits = st.session_state.editor_redes.get('edited_rows', {})
        for row_idx, changes in edits.items():
            for col, val in changes.items():
                st.session_state.df_redes.at[st.session_state.df_redes.index[row_idx], col] = val

    n_arcos = st.session_state.n_arcos_slider
    
    # Solo ajustamos el número de filas (añadimos vacías o recortamos)
    current_len = len(st.session_state.df_redes)
    if n_arcos > current_len:
        # Añadir filas nuevas
        nuevas_filas = pd.DataFrame(
            [["", "", 0.0, 0.0] for _ in range(n_arcos - current_len)],
            columns=st.session_state.df_redes.columns
        )
        st.session_state.df_redes = pd.concat([st.session_state.df_redes, nuevas_filas], ignore_index=True)
    elif n_arcos < current_len:
        # Recortar filas sobrantes
        st.session_state.df_redes = st.session_state.df_redes.iloc[:n_arcos]

# Título principal
st.title("🚀 Suite de Investigación Operativa")
st.markdown("---")

# Menú lateral
with st.sidebar:
    # 1. Cargamos el logo local (Asegúrate de poner la ruta correcta de tu archivo)
    try:
        # Reemplaza 'ruta/a/tu/logo_umsa.png' por el nombre real de tu archivo
        logo_base64 = obtener_base64_de_archivo(r"assets\Logo_Umsa.png") 
        ruta_logo = os.path.join("assets", "Logo_Umsa.png") # Ajusta 'assets' si tu carpeta tiene otro nombre
        if os.path.exists(ruta_logo):
            logo_base64 = obtener_base64_de_archivo(ruta_logo)
            # ... resto de tu lógica de visualización ...
        else:
            st.sidebar.error("No se encontró el archivo del logo en la ruta especificada.")
        
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{logo_base64}" width="100" style="margin-bottom: 10px;">
                <h2 style='color: #00D287; margin-bottom: 0;'>UMSA - FCPN</h2>
                <p style='color: #888; font-size: 0.9rem;'>Investigación Operativa I</p>
            </div>
            <hr style="margin-top: 5px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
    except Exception:
        # Si la imagen no carga, muestra solo el texto para que no de error
        st.markdown("<h2 style='text-align: center; color: #00D287;'>UMSA - FCPN</h2>", unsafe_allow_html=True)
    
    # Menú avanzado con íconos de FontAwesome
    opcion = option_menu(
        menu_title="Módulos",  # Título del menú
        options=["Programación Lineal", "Modelo de Transporte", "Asignación", "Flujo de Redes"],
        icons=["graph-up-arrow", "truck", "people-fill", "diagram-3-fill"], # Íconos
        menu_icon="cast", # Ícono principal del menú
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00D287", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin":"5px", 
                "--hover-color": "#27272A"
            },
            "nav-link-selected": {"background-color": "#27272A", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.caption("👨‍💻 Desarrollado por: Erick")
    st.caption("⚙️ Motor: PuLP + CBC")
# Lógica de navegación
# --- 3. INTERFAZ DE PROGRAMACIÓN LINEAL -------------------------------------------------------------------
if opcion == "Programación Lineal":
    titulo_con_icono("Programación Lineal General", "graph-up-arrow", "h2")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Variables", 1, 10, len(st.session_state.df_obj.columns), 
                    key="n_vars_slider", on_change=ajustar_dimensiones)
        with col2:
            st.slider("Restricciones", 1, 15, len(st.session_state.df_restr), 
                    key="n_cons_slider", on_change=ajustar_dimensiones)
        with col3:
            st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True, key="tipo_opt")

    titulo_con_icono("Coeficientes de la Función Objetivo", "pencil-square", "h3")
    # Mostramos la tabla base. Streamlit se encarga de mostrar los cambios encima.
    st.data_editor(st.session_state.df_obj, key="editor_obj", use_container_width=True)

    titulo_con_icono("Restricciones del Modelo", "list-check", "h3")
    st.data_editor(st.session_state.df_restr, key="editor_restr", use_container_width=True, 
                column_config={
                    "Signo": st.column_config.SelectboxColumn(
                        options=["<=", ">=", "="],
                        required=True,
                    )
                })

    # --- BOTÓN DE RESOLUCIÓN COMPLETO ---
    st.markdown("---")
    if st.button("Resolver con PuLP", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. FUSIÓN DE DATOS (Asegura que el solver lea lo que ves en pantalla)
        # Fusión Función Objetivo
        obj_final = st.session_state.df_obj.copy()
        if 'editor_obj' in st.session_state:
            for r_idx, changes in st.session_state.editor_obj.get('edited_rows', {}).items():
                for c, v in changes.items():
                    obj_final.at[obj_final.index[r_idx], c] = v

        # Fusión Restricciones
        restr_final = st.session_state.df_restr.copy()
        if 'editor_restr' in st.session_state:
            for r_idx, changes in st.session_state.editor_restr.get('edited_rows', {}).items():
                for c, v in changes.items():
                    restr_final.at[restr_final.index[r_idx], c] = v

        # 2. LLAMADA AL SOLVER
        
        resultado = resolver_lp(obj_final, restr_final, st.session_state.tipo_opt)

        # 3. DESPLIEGUE DE RESULTADOS
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"Atención: El problema no es Óptimo. Estado: {resultado['status']}")
        else:
            # Feedback inmediato de éxito
            st.toast("¡Optimización Exitosa!", icon="✅")
            st.balloons()
            
            # --- CREACIÓN DE PESTAÑAS (TABS) MODERNAS ---
            tab_res, tab_graf, tab_exp = st.tabs(["📊 Resultados Numéricos", "📈 Análisis Gráfico", "📄 Exportar PDF"])
            
            # --- PESTAÑA 1: RESULTADOS ---
            with tab_res:
                st.subheader("Métricas Principales")
                c1, c2 = st.columns(2)
                c1.metric("Estado de la Solución", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Valor Óptimo Z", f"{val_z:,.2f}")

                titulo_con_icono("Valores Óptimos de las Variables", "check-circle-fill", "h3")
                df_vars = pd.DataFrame([resultado["variables"]])
                df_vars.index = [1] 
                st.dataframe(df_vars, use_container_width=True)

                titulo_con_icono("Análisis de Sensibilidad", "search", "h3")
                with st.expander("Ver Precios Sombra y Holguras", expanded=True):
                    df_sens = resultado["sensibilidad"].copy()
                    df_sens.index = df_sens.index + 1
                    st.dataframe(df_sens, use_container_width=True)

            # --- PESTAÑA 2: GRÁFICO 2D ---
            with tab_graf:
                if st.session_state.n_vars_slider == 2:
                    titulo_con_icono("Gráfico del Modelo Geometrico", "bar-chart-line", "h3")
                    
                    cols = list(obj_final.columns)
                    x1_name, x2_name = cols[0], cols[1]
                    
                    opt_x1 = resultado["variables"][x1_name]
                    opt_x2 = resultado["variables"][x2_name]
                    
                    max_val = max(10.0, opt_x1 * 1.5, opt_x2 * 1.5)
                    x_vals = np.linspace(0, max_val, 400)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.style.use("dark_background")
                    
                    for idx, row in restr_final.iterrows():
                        c1 = row[x1_name]
                        c2 = row[x2_name]
                        rhs = row['RHS']
                        signo = row['Signo']
                        
                        if c2 != 0:
                            y_vals = (rhs - c1 * x_vals) / c2
                            ax.plot(x_vals, y_vals, label=f"{idx} ({signo} {rhs})", linewidth=2)
                        else:
                            if c1 != 0:
                                x_vert = rhs / c1
                                ax.axvline(x=x_vert, label=f"{idx} ({signo} {rhs})", linewidth=2, color=np.random.rand(3,))

                    ax.plot(opt_x1, opt_x2, marker='*', color='red', markersize=20, label=f"Óptimo ({opt_x1:.2f}, {opt_x2:.2f})")
                    
                    ax.set_xlim(0, max_val)
                    ax.set_ylim(0, max_val)
                    ax.axhline(0, color='white', linewidth=1)
                    ax.axvline(0, color='white', linewidth=1)
                    ax.set_xlabel(f"Variable {x1_name}", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"Variable {x2_name}", fontsize=12, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    st.pyplot(fig)
                else:
                    # Mensaje elegante si no hay 2 variables
                    st.info("💡 El gráfico geométrico solo está disponible para modelos que tengan exactamente 2 variables de decisión.")

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga el reporte detallado con formato institucional para adjuntarlo a tus trabajos.")
                
                ruta_pdf = generar_reporte_pdf(
                    "Programación Lineal", 
                    st.session_state.tipo_opt, 
                    resultado["z"], 
                    resultado["variables"], 
                    resultado.get("sensibilidad")
                )

                with open(ruta_pdf, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte en PDF",
                        data=f,
                        file_name=f"Reporte_Optimización_{st.session_state.tipo_opt}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# --- 3. INTERFAZ DE MODELO DE TRANSPORTE -------------------------------------------------------------------


elif opcion == "Modelo de Transporte":
    titulo_con_icono("Modelo de Transporte", "truck")
    
    # Configuración
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Determinamos cuántos Orígenes hay (restando la fila 'Demanda')
            current_origs = len(st.session_state.df_transp) - 1
            st.slider("Nº de Orígenes (Plantas)", 1, 15, current_origs, 
                      key="n_orig_slider", on_change=ajustar_dimensiones_transp)
        with col2:
            # Determinamos cuántos Destinos hay (restando la columna 'Oferta')
            current_dests = len(st.session_state.df_transp.columns) - 1
            st.slider("Nº de Destinos (Clientes)", 1, 15, current_dests, 
                      key="n_dest_slider", on_change=ajustar_dimensiones_transp)
        with col3:
            st.radio("Objetivo", ["Minimizar", "Maximizar"], horizontal=True, key="tipo_opt_transp")
            
    st.info("💡 **Instrucciones:** Ingresa los costos unitarios de envío en las celdas centrales. Coloca la disponibilidad en la columna **Oferta** y los requerimientos en la fila **Demanda**.")

    # Matriz Interactiva
    titulo_con_icono("Matriz de Costos, Oferta y Demanda", "grid-3x3", "h3")
    st.data_editor(st.session_state.df_transp, key="editor_transp", use_container_width=True)

# --- BOTÓN DE RESOLUCIÓN (Reemplazar en la sección Transporte) ---
    st.markdown("---")
    if st.button("Resolver Modelo de Transporte", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. Consolidación de datos
        df_final_t = st.session_state.df_transp.copy()
        if 'editor_transp' in st.session_state:
            for r_idx, changes in st.session_state.editor_transp.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_t.at[df_final_t.index[r_idx], c] = v
        
        # 2. Llamar al Solver
        from modules.transport_solver import resolver_transporte
        resultado = resolver_transporte(df_final_t, st.session_state.tipo_opt_transp)

        # 3. Mostrar Resultados Modernos
        # 3. Mostrar Resultados Modernos
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"El problema no tiene solución óptima. Estado: {resultado['status']}")
        else:
            # Feedback interactivo
            st.toast("¡Plan de Transporte Óptimo Encontrado!", icon="✅")
            st.balloons()

            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_res, tab_duales, tab_exp = st.tabs(["📊 Matriz de Envíos", "🔄 Análisis Dual", "📄 Exportar PDF"])

            # --- PESTAÑA 1: RESULTADOS Y MATRIZ ---
            with tab_res:
                # Avisos de IO1: Balanceo
                if not resultado["balanceado"]:
                    st.info("ℹ️ **Nota Académica:** El modelo original estaba **Desbalanceado**. El programa creó un nodo ficticio automáticamente para absorber la diferencia sin afectar el costo.")
                
                # Métricas
                c1, c2 = st.columns(2)
                c1.metric("Estado", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Costo/Beneficio Óptimo (Z)", f"{val_z:,.2f}")

                # Matriz de Resultados
                titulo_con_icono("Matriz Óptima de Envíos", "box-seam", "h3")
                st.write("Las celdas muestran la cantidad que debes enviar desde cada Origen a cada Destino.")
                st.dataframe(resultado["matriz_resultados"], use_container_width=True)

            # --- PESTAÑA 2: DUALES ---
            with tab_duales:
                titulo_con_icono("Multiplicadores Duales", "arrow-repeat", "h3")
                st.write("Valores duales ($u_i, v_j$) asociados a las restricciones de oferta y demanda del modelo:")
                
                with st.container(border=True):
                    c_dual_o, c_dual_d = st.columns(2)
                    with c_dual_o:
                        st.markdown("**Orígenes ($u_i$)**")
                        st.json(resultado["duales_oferta"])
                    with c_dual_d:
                        st.markdown("**Destinos ($v_j$)**")
                        st.json(resultado["duales_demanda"])

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga el reporte detallado con la matriz de envíos y análisis de balanceo.")
                
                # Preparamos el PDF de Transporte
                ruta_pdf_t = generar_reporte_pdf_transporte(
                    st.session_state.tipo_opt_transp,
                    resultado["z"],
                    resultado["matriz_resultados"],
                    resultado["duales_oferta"],
                    resultado["duales_demanda"],
                    resultado["balanceado"]
                )

                with open(ruta_pdf_t, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Transporte (PDF)",
                        data=f,
                        file_name=f"Reporte_Transporte_{st.session_state.tipo_opt_transp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )




# --- MÓDULO: MODELO DE ASIGNACIÓN -----------------------------------------------------------------
elif opcion == "Asignación":
    titulo_con_icono("Modelo de Asignación", "people-fill")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Nº de Trabajadores / Agentes", 1, 15, len(st.session_state.df_asign.index), 
                      key="n_workers_slider", on_change=ajustar_dimensiones_asign)
        with col2:
            st.slider("Nº de Tareas / Destinos", 1, 15, len(st.session_state.df_asign.columns), 
                      key="n_tasks_slider", on_change=ajustar_dimensiones_asign)
        with col3:
            st.radio("Objetivo", ["Minimizar", "Maximizar"], horizontal=True, key="tipo_opt_asign")
            
    st.info("💡 **Instrucciones:** Ingresa los costos, tiempos o utilidades en la matriz. El programa balanceará automáticamente si el número de trabajadores no coincide con el de tareas.")

    titulo_con_icono("Matriz de Asignación", "table", "h3")
    st.data_editor(st.session_state.df_asign, key="editor_asign", use_container_width=True)

    st.markdown("---")
    if st.button("Resolver Asignación", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # Consolidar datos
        df_final_a = st.session_state.df_asign.copy()
        if 'editor_asign' in st.session_state:
            for r_idx, changes in st.session_state.editor_asign.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_a.at[df_final_a.index[r_idx], c] = v
        
        # Llamar al Solver
        from modules.assignment_solver import resolver_asignacion
        resultado = resolver_asignacion(df_final_a, st.session_state.tipo_opt_asign)

        # Mostrar Resultados
        # Mostrar Resultados
        if "error" in resultado:
            st.error(f"Error técnico: {resultado['error']}")
        elif resultado["status"] != "Optimal":
            st.warning(f"No se encontró solución óptima. Estado: {resultado['status']}")
        else:
            # Feedback interactivo
            st.toast("¡Asignación Óptima Completada!", icon="✅")
            st.balloons()

            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_plan, tab_exp = st.tabs(["🎯 Plan de Asignación", "📄 Exportar PDF"])

            # --- PESTAÑA 1: PLAN Y MATRIZ ---
            with tab_plan:
                if not resultado["balanceado"]:
                    st.info(f"ℹ️ **Nota de Balanceo:** Había {resultado['n_trabajadores']} trabajadores y {resultado['n_tareas']} tareas. Se añadieron elementos ficticios automáticamente.")

                c1, c2 = st.columns(2)
                c1.metric("Estado", resultado["status"])
                val_z = resultado.get("z", 0.0)
                c2.metric("Valor Óptimo (Z)", f"{val_z:,.2f}")

                titulo_con_icono("Matriz de Asignación", "person-badge", "h3")
                st.write("A continuación se muestra la asignación óptima de tareas:")
                st.table(resultado["lista_asignaciones"])
                
                with st.expander("Ver Matriz Binaria de Decisión ($x_{ij}$)", expanded=False):
                    st.write("Representación matemática de la asignación (1 = Asignado, 0 = No asignado):")
                    st.dataframe(resultado["matriz_resultados"], use_container_width=True)

            # --- PESTAÑA 2: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Obtén un reporte formal en PDF con el plan de asignación y la matriz técnica.")
                
                # Preparamos el PDF de Asignación
                ruta_pdf_a = generar_reporte_pdf_asignacion(
                    st.session_state.tipo_opt_asign,
                    resultado["z"],
                    resultado["lista_asignaciones"],
                    resultado["matriz_resultados"],
                    resultado["balanceado"]
                )

                with open(ruta_pdf_a, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Asignación (PDF)",
                        data=f,
                        file_name=f"Reporte_Asignacion_{st.session_state.tipo_opt_asign}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )












# --- MÓDULO: FLUJO DE REDES ---
elif opcion == "Flujo de Redes":
    titulo_con_icono("Flujo de Redes", "diagram-3-fill")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.slider("Número de Arcos (Rutas)", 1, 30, len(st.session_state.df_redes), 
                      key="n_arcos_slider", on_change=ajustar_dimensiones_redes)
        with col2:
            st.radio(
                "Tipo de Problema", 
                ["Flujo Máximo", "Ruta Más Corta"], 
                horizontal=True, 
                key="tipo_problema_redes"
            )
            
    st.info("💡 **Instrucciones:** Define tu grafo. Usa letras o números para los nodos (ej. 'O' para Origen, 'T' para Destino).")

    # Inputs de Nodos Inicio y Fin
    c_origen, c_destino = st.columns(2)
    with c_origen:
        st.text_input("📍 Nodo de Inicio (Source)", value="O", key="nodo_inicio")
    with c_destino:
        st.text_input("🏁 Nodo Final (Sink)", value="T", key="nodo_final")

    titulo_con_icono("Definición de Arcos", "vector-pen", "h3")
    
    tipo_prob = st.session_state.tipo_problema_redes
    column_config = {
        "Nodo Origen": st.column_config.TextColumn("Nodo Origen", required=True),
        "Nodo Destino": st.column_config.TextColumn("Nodo Destino", required=True),
    }
    
    # Ocultar columnas irrelevantes
    if tipo_prob == "Flujo Máximo":
        column_config["Costo Unitario"] = None
    elif tipo_prob == "Ruta Más Corta":
        column_config["Capacidad"] = None
        
    st.data_editor(
        st.session_state.df_redes, 
        key="editor_redes", 
        use_container_width=True,
        column_config=column_config,
        hide_index=True
    )

    st.markdown("---")
    if st.button("Resolver Red de Nodos", type="primary", icon=":material/rocket_launch:", use_container_width=True):
        
        # 1. Consolidación de datos
        df_final_r = st.session_state.df_redes.copy()
        if 'editor_redes' in st.session_state:
            for r_idx, changes in st.session_state.editor_redes.get('edited_rows', {}).items():
                for c, v in changes.items():
                    df_final_r.at[df_final_r.index[r_idx], c] = v

        from modules.network_solver import resolver_flujo_maximo, resolver_ruta_mas_corta
        origen = st.session_state.nodo_inicio.strip()
        destino = st.session_state.nodo_final.strip()

        # 2. Enrutador
        if tipo_prob == "Flujo Máximo":
            resultado = resolver_flujo_maximo(df_final_r, origen, destino)
            label_z = "Flujo Máximo Total"
        else: # Ruta Más Corta
            resultado = resolver_ruta_mas_corta(df_final_r, origen, destino)
            label_z = "Distancia Mínima Total"

        # 3. Mostrar Resultados
        if "error" in resultado:
            st.error(resultado["error"])
        elif resultado["status"] != "Optimal":
            st.warning("No se encontró una solución óptima para esta red.")
        else:
            # Feedback interactivo
            st.toast(f"¡Cálculo de {tipo_prob} Exitoso!", icon="✅")
            st.balloons()
            
            # --- CREACIÓN DE PESTAÑAS (TABS) ---
            tab_res, tab_graf, tab_exp = st.tabs(["🛤️ Resultados Numéricos", "🌐 Grafo Visual", "📄 Exportar PDF"])

            # --- PESTAÑA 1: RESULTADOS ---
            with tab_res:
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Estado", "Óptimo")
                c2.metric(label_z, f"{resultado['valor']:,.2f}")
                
                titulo_con_icono("Detalle de Rutas Utilizadas", "geo-alt", "h3")
                if not resultado["rutas"].empty:
                    st.dataframe(resultado["rutas"], use_container_width=True)
                    
                    # Análisis de cuellos de botella exclusivo para Flujo Máximo
                    if tipo_prob == "Flujo Máximo":
                        df_rutas = resultado["rutas"]
                        rutas_saturadas = len(df_rutas[df_rutas["Capacidad Sobrante"] == 0])
                        st.info(f"💡 **Análisis de Cuellos de Botella:** Tienes **{rutas_saturadas}** rutas operando a su máxima capacidad (Sobrante = 0). Para mejorar el flujo total, debes expandir la capacidad de estas rutas.")
                else:
                    st.warning("No es posible establecer una ruta con esta configuración.")

            # --- PESTAÑA 2: VISUALIZACIÓN GRÁFICA ---
            with tab_graf:
                titulo_con_icono("Visualización del Grafo Óptimo", "share", "h3")
                st.write("Las rutas resaltadas en verde representan el flujo óptimo o la ruta más corta encontrada.")
                
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.style.use("dark_background") # Estilo oscuro para proteger la vista
                    
                    G = nx.DiGraph() 
                    
                    rutas_activas = []
                    for _, row in resultado["rutas"].iterrows():
                        rutas_activas.append((str(row["De"]), str(row["A"])))
                    
                    for _, row in df_final_r.iterrows():
                        u = str(row['Nodo Origen']).strip()
                        v = str(row['Nodo Destino']).strip()
                        if u and v: 
                            G.add_edge(u, v)
                    
                    pos = nx.spring_layout(G, seed=42) 
                    
                    nx.draw_networkx_nodes(G, pos, node_color='#9b59b6', node_size=700, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)
                    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='#555555', arrows=True, ax=ax)
                    nx.draw_networkx_edges(G, pos, edgelist=rutas_activas, edge_color='#2ecc71', width=3.0, arrows=True, arrowsize=20, ax=ax)
                    
                    ax.set_title(f"Grafo de Solución: {tipo_prob}", color='white', pad=20)
                    plt.axis('off') 
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"No se pudo generar la visualización gráfica del grafo: {e}")

            # --- PESTAÑA 3: EXPORTACIÓN ---
            with tab_exp:
                titulo_con_icono("Reporte y Documentación", "file-earmark-pdf-fill", "h3")
                st.write("Descarga un reporte formal en PDF con las rutas utilizadas y métricas generales del modelo de redes.")
                
                ruta_pdf_r = generar_reporte_pdf_redes(
                    tipo_prob,
                    origen,
                    destino,
                    resultado["valor"],
                    resultado["rutas"]
                )

                with open(ruta_pdf_r, "rb") as f:
                    st.download_button(
                        label="📥 Descargar Reporte de Redes (PDF)",
                        data=f,
                        file_name=f"Reporte_{tipo_prob.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )