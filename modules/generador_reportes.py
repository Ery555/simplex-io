from weasyprint import HTML
import tempfile

def generar_reporte_pdf(tipo_mod, opt, z, vars_dict, sensibil_df=None):
    """Genera un reporte profesional en formato PDF."""
    
    # Estilos CSS para el PDF
    css_estilos = """
    @page {
        size: A4;
        margin: 20mm;
        background-color: #ffffff;
    }
    body {
        font-family: 'Helvetica', sans-serif;
        color: #333;
        line-height: 1.6;
    }
    .header {
        text-align: center;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h1 { color: #2c3e50; margin: 5px 0; font-size: 18pt; }
    h2 { color: #34495e; border-left: 5px solid #3498db; padding-left: 10px; font-size: 14pt; margin-top: 30px; }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    th, td {
        border: 1px solid #bdc3c7;
        padding: 8px;
        text-align: left;
        font-size: 10pt;
    }
    th { background-color: #ecf0f1; color: #2c3e50; }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 9pt;
        color: #7f8c8d;
        border-top: 1px solid #eee;
        padding-top: 10px;
    }
    """

    # Construcción de las tablas de variables
    filas_vars = "".join([f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in vars_dict.items()])
    
    # Construcción de la tabla de sensibilidad (si existe)
    tabla_sensibilidad = ""
    if sensibil_df is not None:
        encabezados = "".join([f"<th>{col}</th>" for col in sensibil_df.columns])
        filas_s = ""
        for _, row in sensibil_df.iterrows():
            filas_s += "<tr>" + "".join([f"<td>{val}</td>" for val in row]) + "</tr>"
        
        tabla_sensibilidad = f"""
        <h2>Análisis de Sensibilidad / Dualidad</h2>
        <table>
            <thead><tr>{encabezados}</tr></thead>
            <tbody>{filas_s}</tbody>
        </table>
        """

    # HTML Completo
    html_content = f"""
    <html>
    <head><style>{css_estilos}</style></head>
    <body>
        <div class="header">
            <p>UNIVERSIDAD MAYOR DE SAN ANDRÉS</p>
            <p>FACULTAD DE CIENCIAS PURAS Y NATURALES</p>
            <h1>Reporte de Optimización: {tipo_mod}</h1>
        </div>

        <div class="metric-box">
            <p><strong>Tipo de Objetivo:</strong> {opt}</p>
            <p><strong>Valor Óptimo Alcanzado (Z):</strong> <span style="font-size: 14pt; color: #27ae60;">{z:,.4f}</span></p>
        </div>

        <h2>Variables de Decisión</h2>
        <table>
            <thead><tr><th>Variable</th><th>Valor Óptimo</th></tr></thead>
            <tbody>{filas_vars}</tbody>
        </table>

        {tabla_sensibilidad}

        <div class="footer">
            <p>Reporte generado automáticamente por la Suite de Investigación Operativa</p>
            <p><strong>Autor: Erick</strong></p>
            <p>La Paz - Bolivia</p>
        </div>
    </body>
    </html>
    """

    # Generación del PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        HTML(string=html_content).write_pdf(tmp.name)
        return tmp.name
    

def generar_reporte_pdf_transporte(opt, z, df_res, dual_o, dual_d, es_bal):
    """Genera un reporte PDF específico para el Modelo de Transporte."""
    
    css_estilos = """
    @page { size: A4; margin: 20mm; }
    body { font-family: 'Helvetica', sans-serif; color: #333; }
    .header { text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }
    h1 { color: #2c3e50; font-size: 18pt; margin: 5px 0; }
    h2 { color: #34495e; border-left: 5px solid #e67e22; padding-left: 10px; font-size: 14pt; margin-top: 25px; }
    .metric-box { background-color: #fffaf0; padding: 15px; border-radius: 8px; border: 1px solid #f39c12; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }
    th, td { border: 1px solid #bdc3c7; padding: 6px; text-align: center; font-size: 9pt; word-wrap: break-word; }
    th { background-color: #f39c12; color: white; }
    .footer { margin-top: 40px; text-align: center; font-size: 9pt; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 10px; }
    """

    # Construcción dinámica de la Matriz de Envíos en HTML
    # Encabezados (Destinos)
    header_html = "<th>Origen \ Destino</th>" + "".join([f"<th>{col}</th>" for col in df_res.columns])
    
    # Filas (Orígenes + Valores)
    rows_html = ""
    for idx, row in df_res.iterrows():
        rows_html += f"<tr><td style='font-weight:bold; background-color:#f9f9f9;'>{idx}</td>"
        rows_html += "".join([f"<td>{val:,.2f}</td>" if val > 0 else "<td style='color:#ccc;'>0</td>" for val in row])
        rows_html += "</tr>"

    # Tablas de Multiplicadores Duales
    duales_o_html = "".join([f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in dual_o.items()])
    duales_d_html = "".join([f"<tr><td>{k}</td><td>{v:,.4f}</td></tr>" for k, v in dual_d.items()])

    html_content = f"""
    <html>
    <head><style>{css_estilos}</style></head>
    <body>
        <div class="header">
            <p>UNIVERSIDAD MAYOR DE SAN ANDRÉS</p>
            <p>FACULTAD DE CIENCIAS PURAS Y NATURALES</p>
            <h1>Reporte: Modelo de Transporte</h1>
        </div>

        <div class="metric-box">
            <p><strong>Objetivo:</strong> {opt}</p>
            <p><strong>Costo Total Óptimo (Z):</strong> <span style="font-size: 14pt; color: #d35400;">{z:,.2f}</span></p>
            <p><strong>Estado de Balanceo:</strong> {'Balanceado' if es_bal else 'Re-balanceado con Nodos Ficticios'}</p>
        </div>

        <h2>Matriz Óptima de Envíos (xᵢⱼ)</h2>
        <table>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>

        <div style="display: flex; justify-content: space-between;">
            <div style="width: 48%; float: left;">
                <h2>Duales Origen (uᵢ)</h2>
                <table><thead><tr><th>Nodo</th><th>Valor</th></tr></thead><tbody>{duales_o_html}</tbody></table>
            </div>
            <div style="width: 48%; float: right;">
                <h2>Duales Destino (vⱼ)</h2>
                <table><thead><tr><th>Nodo</th><th>Valor</th></tr></thead><tbody>{duales_d_html}</tbody></table>
            </div>
        </div>
        <div style="clear: both;"></div>

        <div class="footer">
            <p>Reporte generado por la Suite de Investigación Operativa</p>
            <p><strong>Autor: Erick</strong></p>
            <p>La Paz - Bolivia</p>
        </div>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        HTML(string=html_content).write_pdf(tmp.name)
        return tmp.name
    

def generar_reporte_pdf_asignacion(opt, z, df_lista, df_matriz, es_bal):
    """Genera un reporte PDF profesional para el Modelo de Asignación con matriz compacta."""
    
    # 1. Estilos completos fusionados
    css_estilos = """
    @page { size: A4 landscape; margin: 15mm; }
    body { font-family: 'Helvetica', sans-serif; color: #333; }
    .header { text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }
    h1 { color: #2c3e50; font-size: 18pt; margin: 5px 0; }
    h2 { color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; font-size: 14pt; margin-top: 25px; }
    .metric-box { background-color: #ebf5fb; padding: 15px; border-radius: 8px; border: 1px solid #3498db; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: center; font-size: 10pt; }
    th { background-color: #3498db; color: white; }
    .highlight { font-weight: bold; color: #2980b9; }
    .footer { margin-top: 50px; text-align: center; font-size: 9pt; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 10px; }

    /* ESTILOS NUEVOS PARA LA MATRIZ COMPACTA */
    .tabla-binaria {
        table-layout: auto !important; 
        width: 100% !important;
        margin: 10px 0;
    }
    .tabla-binaria th, .tabla-binaria td {
        padding: 4px 2px !important;  /* Relleno mínimo para ganar espacio */
        font-size: 8pt !important;    /* Fuente reducida */
        white-space: nowrap;          /* Evita cortes de palabras */
    }
    """

    # 2. Tabla de Lista de Asignaciones (Principal)
    filas_lista = ""
    for _, row in df_lista.iterrows():
        filas_lista += f"""
        <tr>
            <td>{row['Trabajador / Agente']}</td>
            <td class="highlight">{row['Tarea Asignada']}</td>
            <td>{row['Costo / Aporte']:.2f}</td>
        </tr>
        """

    limite_columnas = 10  # Límite seguro para hoja horizontal
    
    # Usamos .shape[1] que es infalible para contar columnas en Pandas
    if df_matriz.shape[1] <= limite_columnas:
        header_matriz = "<th>Trab / Tarea</th>" + "".join([f"<th>{str(col)[:8]}..</th>" if len(str(col)) > 10 else f"<th>{col}</th>" for col in df_matriz.columns])
        rows_matriz = ""
        for idx, row in df_matriz.iterrows():
            nombre_trab = str(idx)[:12] + ".." if len(str(idx)) > 14 else str(idx)
            rows_matriz += f"<tr><td style='font-weight:bold; background-color:#f4f4f4;'>{nombre_trab}</td>"
            rows_matriz += "".join([f"<td style='color:{'#27ae60' if val > 0.5 else '#ccc'}; font-weight:{'bold' if val > 0.5 else 'normal'};'>{'1' if val > 0.5 else '0'}</td>" for val in row])
            rows_matriz += "</tr>"
            
        seccion_matriz_html = f"""
        <h2>Matriz Binaria de Decisión (xᵢⱼ)</h2>
        <table class="tabla-binaria">
            <thead><tr>{header_matriz}</tr></thead>
            <tbody>{rows_matriz}</tbody>
        </table>
        """
    else:
        seccion_matriz_html = f"""
        <h2>Matriz Binaria de Decisión (xᵢⱼ)</h2>
        <div style="background-color: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; color: #856404; font-size: 10pt; margin-top: 10px;">
            <strong>Nota del Sistema:</strong> Debido a la alta dimensionalidad del problema ({df_matriz.shape[0]}x{df_matriz.shape[1]}), la matriz binaria ha sido omitida de este reporte para evitar el desbordamiento de la página. Por favor, consulte la tabla de <em>Plan de Asignación Óptima</em>.
        </div>
        """

    # 4. HTML Completo
    html_content = f"""
    <html>
    <head><style>{css_estilos}</style></head>
    <body>
        <div class="header">
            <p>UNIVERSIDAD MAYOR DE SAN ANDRÉS</p>
            <p>FACULTAD DE CIENCIAS PURAS Y NATURALES</p>
            <h1>Reporte: Modelo de Asignación</h1>
        </div>

        <div class="metric-box">
            <p><strong>Objetivo del Modelo:</strong> {opt}</p>
            <p><strong>Valor Óptimo de la Función (Z):</strong> <span style="font-size: 14pt; color: #2980b9;">{z:,.2f}</span></p>
            <p><strong>Balanceo:</strong> {'El modelo estaba cuadrado' if es_bal else 'Se aplicó balanceo con elementos ficticios'}</p>
        </div>

        <h2>Resultados: Plan de Asignación Óptima</h2>
        <table>
            <thead>
                <tr>
                    <th>Trabajador / Agente</th>
                    <th>Tarea Asignada</th>
                    <th>Costo / Aporte</th>
                </tr>
            </thead>
            <tbody>
                {filas_lista}
            </tbody>
        </table>

        {seccion_matriz_html}

        <div class="footer">
            <p>Reporte generado por la Suite de Investigación Operativa</p>
            <p><strong>Autor: Erick</strong></p>
            <p>La Paz - Bolivia</p>
        </div>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        HTML(string=html_content).write_pdf(tmp.name)
        return tmp.name
    


def generar_reporte_pdf_redes(tipo_prob, origen, destino, z, df_rutas):
    """Genera un reporte PDF profesional para el Modelo de Redes."""
    
    css_estilos = """
    @page { size: A4; margin: 20mm; }
    body { font-family: 'Helvetica', sans-serif; color: #333; }
    .header { text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }
    h1 { color: #2c3e50; font-size: 18pt; margin: 5px 0; }
    h2 { color: #8e44ad; border-left: 5px solid #8e44ad; padding-left: 10px; font-size: 14pt; margin-top: 25px; }
    .metric-box { background-color: #f5eef8; padding: 15px; border-radius: 8px; border: 1px solid #9b59b6; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: center; font-size: 10pt; }
    th { background-color: #9b59b6; color: white; }
    .highlight { font-weight: bold; color: #8e44ad; }
    .footer { margin-top: 50px; text-align: center; font-size: 9pt; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 10px; }
    """

    # Construir filas de la tabla según el tipo de problema
    encabezados = "".join([f"<th>{col}</th>" for col in df_rutas.columns])
    filas = ""
    for _, row in df_rutas.iterrows():
        filas += "<tr>" + "".join([f"<td>{val}</td>" if isinstance(val, str) else f"<td>{val:,.2f}</td>" for val in row]) + "</tr>"

    html_content = f"""
    <html>
    <head><style>{css_estilos}</style></head>
    <body>
        <div class="header">
            <p>UNIVERSIDAD MAYOR DE SAN ANDRÉS</p>
            <p>FACULTAD DE CIENCIAS PURAS Y NATURALES</p>
            <h1>Reporte: Optimización de Redes</h1>
        </div>

        <div class="metric-box">
            <p><strong>Tipo de Problema:</strong> {tipo_prob}</p>
            <p><strong>Ruta Evaluada:</strong> Desde el nodo <span class="highlight">'{origen}'</span> hasta el nodo <span class="highlight">'{destino}'</span></p>
            <p><strong>Valor Óptimo Total:</strong> <span style="font-size: 14pt; color: #8e44ad;">{z:,.2f}</span></p>
        </div>

        <h2>Detalle de Arcos (Rutas) Utilizados</h2>
        <table>
            <thead><tr>{encabezados}</tr></thead>
            <tbody>{filas}</tbody>
        </table>

        <div class="footer">
            <p>Reporte generado por la Suite de Investigación Operativa</p>
            <p><strong>Autor: Erick</strong></p>
            <p>La Paz - Bolivia</p>
        </div>
    </body>
    </html>
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        HTML(string=html_content).write_pdf(tmp.name)
        return tmp.name