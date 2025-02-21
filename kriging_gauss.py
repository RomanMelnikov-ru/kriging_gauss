import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from skgstat import Variogram
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist
import ezdxf
import matplotlib.pyplot as plt
import io
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация состояния сессии
if 'x' not in st.session_state:
    st.session_state.x = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'z' not in st.session_state:
    st.session_state.z = None
if 'V' not in st.session_state:
    st.session_state.V = None
if 'nugget' not in st.session_state:
    st.session_state.nugget = None
if 'sill' not in st.session_state:
    st.session_state.sill = None
if 'range_' not in st.session_state:
    st.session_state.range_ = None
if 'grid_x' not in st.session_state:
    st.session_state.grid_x = None
if 'grid_y' not in st.session_state:
    st.session_state.grid_y = None
if 'z_pred' not in st.session_state:
    st.session_state.z_pred = None
if 'sigma' not in st.session_state:
    st.session_state.sigma = None
if 'grid_size' not in st.session_state:
    st.session_state.grid_size = None
if 'padding' not in st.session_state:
    st.session_state.padding = None


# Функция для загрузки данных из Excel
def load_data():
    uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            if 'X' not in data.columns or 'Y' not in data.columns or 'Z' not in data.columns:
                st.error("Файл должен содержать столбцы X, Y и Z.")
                return
            st.session_state.x = data['X'].values
            st.session_state.y = data['Y'].values
            st.session_state.z = data['Z'].values

            # Вывод информации о данных
            num_points = len(st.session_state.x)
            distances = pdist(np.vstack((st.session_state.x, st.session_state.y)).T)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            min_z = np.min(st.session_state.z)
            max_z = np.max(st.session_state.z)

            st.write("Информация о данных:")
            st.table(pd.DataFrame({
                "Параметр": ["Количество точек", "Минимальное расстояние", "Максимальное расстояние", "Минимальное Z",
                             "Максимальное Z"],
                "Значение": [num_points, f"{min_distance:.2f}", f"{max_distance:.2f}", f"{min_z:.2f}", f"{max_z:.2f}"]
            }))
        except Exception as e:
            st.error(f"Не удалось загрузить данные: {str(e)}")


# Функция для построения эмпирической и теоретической вариограммы
def plot_empirical_variogram():
    if st.session_state.x is None:
        st.error("Сначала загрузите данные!")
        return
    try:
        st.session_state.V = Variogram(coordinates=np.vstack((st.session_state.x, st.session_state.y)).T,
                                       values=st.session_state.z, model='gaussian')
        range_, sill, nugget = st.session_state.V.parameters
        st.session_state.range_ = range_
        st.session_state.sill = round(sill, 3)  # Округление sill до 3 знаков
        st.session_state.nugget = nugget

        # Построение графика
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.V.bins, y=st.session_state.V.experimental, mode='markers',
                                 name='Экспериментальная вариограмма'))
        h = np.linspace(0, np.max(st.session_state.V.bins), 100)
        theoretical = nugget + (sill - nugget) * (1 - np.exp(-(h ** 2) / (range_ ** 2)))
        fig.add_trace(go.Scatter(x=h, y=theoretical, mode='lines', name='Теоретическая вариограмма (модель Гаусса)'))
        fig.update_layout(title='Экспериментальная и теоретическая вариограмма',
                          xaxis_title='Расстояние (h)',
                          yaxis_title='Полудисперсия (γ(h))')
        st.plotly_chart(fig)

        # Вывод параметров в таблицу
        st.write("Параметры вариограммы:")
        st.table(pd.DataFrame({
            "Параметр": ["Range (Диапазон)", "Sill (Силл)", "Nugget (Нугет)", "Модель"],
            "Значение": [f"{range_:.2f}", f"{st.session_state.sill:.3f}", f"{nugget:.2f}", "Гаусс"]
        }))
    except Exception as e:
        st.error(f"Не удалось построить вариограмму: {str(e)}")


# Функция для редактирования теоретической вариограммы
def edit_variogram():
    if st.session_state.x is None:
        st.error("Сначала загрузите данные!")
        return
    try:
        st.write("Исходные параметры вариограммы:")
        st.table(pd.DataFrame({
            "Параметр": ["Range (Диапазон)", "Sill (Силл)", "Nugget (Нугет)", "Модель"],
            "Значение": [f"{st.session_state.range_:.2f}", f"{st.session_state.sill:.3f}",
                         f"{st.session_state.nugget:.2f}", "Гаусс"]
        }))

        # Ввод новых значений
        new_range = st.number_input("Введите новое значение Range:", value=st.session_state.range_)
        new_sill = st.number_input("Введите новое значение Sill:", value=st.session_state.sill,
                                   format="%.3f")  # Формат с 3 знаками
        new_nugget = st.number_input("Введите новое значение Nugget:", value=st.session_state.nugget)

        if new_sill <= new_nugget:
            st.error("Sill должен быть больше Nugget.")
            return

        # Построение обновленной вариограммы
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.V.bins, y=st.session_state.V.experimental, mode='markers',
                                 name='Экспериментальная вариограмма'))
        h = np.linspace(0, np.max(st.session_state.V.bins), 100)
        original_theoretical = st.session_state.nugget + (st.session_state.sill - st.session_state.nugget) * (
                    1 - np.exp(-(h ** 2) / (st.session_state.range_ ** 2)))
        fig.add_trace(go.Scatter(x=h, y=original_theoretical, mode='lines', name='Исходная теоретическая вариограмма'))
        updated_theoretical = new_nugget + (new_sill - new_nugget) * (1 - np.exp(-(h ** 2) / (new_range ** 2)))
        fig.add_trace(
            go.Scatter(x=h, y=updated_theoretical, mode='lines', name='Обновлённая теоретическая вариограмма'))
        fig.update_layout(title='Сравнение исходной и обновлённой теоретической вариограммы',
                          xaxis_title='Расстояние (h)',
                          yaxis_title='Полудисперсия (γ(h))')
        st.plotly_chart(fig)

        # Обновление параметров
        st.session_state.range_ = new_range
        st.session_state.sill = round(new_sill, 3)  # Округление sill до 3 знаков
        st.session_state.nugget = new_nugget
    except Exception as e:
        st.error(f"Не удалось обновить вариограмму: {str(e)}")


# Функция для выполнения кригинга
def run_kriging():
    if st.session_state.x is None:
        st.error("Сначала загрузите данные!")
        return
    if st.session_state.nugget is None or st.session_state.sill is None or st.session_state.range_ is None:
        st.error("Сначала постройте вариограмму!")
        return
    try:
        st.write("Параметры вариограммы для кригинга:")
        st.table(pd.DataFrame({
            "Параметр": ["Range (Диапазон)", "Sill (Силл)", "Nugget (Нугет)", "Модель"],
            "Значение": [f"{st.session_state.range_:.2f}", f"{st.session_state.sill:.3f}",
                         f"{st.session_state.nugget:.2f}", "Гауссова"]
        }))

        # Сохраняем grid_size и padding в st.session_state
        st.session_state.grid_size = st.number_input("Количество точек сетки:", min_value=10, value=100)
        st.session_state.padding = st.number_input("Величина отступа:", value=0.0)

        if st.button("Выполнить кригинг"):
            st.session_state.grid_x = np.linspace(min(st.session_state.x) - st.session_state.padding,
                                                  max(st.session_state.x) + st.session_state.padding,
                                                  st.session_state.grid_size)
            st.session_state.grid_y = np.linspace(min(st.session_state.y) - st.session_state.padding,
                                                  max(st.session_state.y) + st.session_state.padding,
                                                  st.session_state.grid_size)
            OK = OrdinaryKriging(
                st.session_state.x, st.session_state.y, st.session_state.z,
                variogram_model='gaussian',
                variogram_parameters={'sill': st.session_state.sill, 'range': st.session_state.range_,
                                      'nugget': st.session_state.nugget},
                nlags=10
            )
            progress_bar = st.progress(0)
            st.session_state.z_pred, st.session_state.sigma = OK.execute('grid', st.session_state.grid_x,
                                                                         st.session_state.grid_y)
            progress_bar.progress(100)

            # Изолинии кригинга с исходными точками
            fig_contour = go.Figure(
                data=go.Contour(z=st.session_state.z_pred, x=st.session_state.grid_x, y=st.session_state.grid_y,
                                colorscale='Viridis'))
            fig_contour.add_trace(
                go.Scatter(x=st.session_state.x, y=st.session_state.y, mode='markers', name='Исходные точки',
                           marker=dict(color='red', size=4)))
            fig_contour.update_layout(
                title='Изополя после кригинга',
                xaxis_title='X',
                yaxis_title='Y',
                autosize=False,
                width=800,
                height=800,
                xaxis=dict(scaleanchor="y", scaleratio=1),  # Равный масштаб X и Y
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            st.plotly_chart(fig_contour)

            # 3D поверхность с aspectmode='data' и исходными точками
            fig_3d = go.Figure(
                data=[go.Surface(z=st.session_state.z_pred, x=st.session_state.grid_x, y=st.session_state.grid_y)])
            fig_3d.add_trace(
                go.Scatter3d(x=st.session_state.x, y=st.session_state.y, z=st.session_state.z, mode='markers',
                             name='Исходные точки', marker=dict(color='red', size=3)))
            fig_3d.update_layout(title='3D поверхность после кригинга', scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # Режим пропорционального отображения
            ))
            st.plotly_chart(fig_3d)
    except Exception as e:
        st.error(f"Не удалось выполнить кригинг: {str(e)}")


# Функция для создания DXF-файла в памяти
def create_dxf_file(grid_x, grid_y, z_pred, step):
    try:
        # Создаем новый DXF-документ
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Добавляем границы
        boundary_points = [
            (min(grid_x), min(grid_y), 0),
            (max(grid_x), min(grid_y), 0),
            (max(grid_x), max(grid_y), 0),
            (min(grid_x), max(grid_y), 0),
            (min(grid_x), min(grid_y), 0)
        ]
        msp.add_polyline3d(boundary_points)

        # Создаем изополи вручную
        min_z_pred = np.min(z_pred)
        max_z_pred = np.max(z_pred)
        levels = np.arange(min_z_pred, max_z_pred, step)

        for level in levels:
            # Находим точки, где z_pred близко к level
            contours = []
            for i in range(len(grid_x) - 1):
                for j in range(len(grid_y) - 1):
                    z1 = z_pred[i][j]
                    z2 = z_pred[i + 1][j]
                    z3 = z_pred[i + 1][j + 1]
                    z4 = z_pred[i][j + 1]

                    # Проверяем, пересекает ли уровень текущий квадрат
                    if (z1 <= level <= z2) or (z1 <= level <= z3) or (z1 <= level <= z4) or \
                       (z2 <= level <= z1) or (z2 <= level <= z3) or (z2 <= level <= z4) or \
                       (z3 <= level <= z1) or (z3 <= level <= z2) or (z3 <= level <= z4) or \
                       (z4 <= level <= z1) or (z4 <= level <= z2) or (z4 <= level <= z3):
                        # Добавляем точки контура
                        contours.append((grid_x[i], grid_y[j], level))

            if contours:
                msp.add_polyline3d(contours)

        # Сохраняем DXF-документ в BytesIO
        output = io.BytesIO()
        doc.saveas(output)
        output.seek(0)  # Перематываем поток в начало
        return output
    except Exception as e:
        st.error(f"Ошибка при создании DXF-файла: {e}")
        return None


# Функция для сохранения результатов
def save_results():
    if st.session_state.z_pred is None:
        st.error("Сначала выполните кригинг!")
        return
    try:
        # Общее количество точек
        total_points = len(st.session_state.grid_x) * len(st.session_state.grid_y)
        st.info(f"Общее количество рассчитанных точек: {total_points}")

        # Пользовательское значение для уменьшения плотности точек
        target_points = st.number_input("Укажите желаемое количество точек для сохранения:", min_value=1,
                                        max_value=total_points, value=total_points)

        # Кнопка для сброса до исходного количества точек
        if st.button("Сбросить до исходного количества точек"):
            # Пересчет grid_x, grid_y и z_pred
            st.session_state.grid_x = np.linspace(min(st.session_state.x) - st.session_state.padding,
                                                  max(st.session_state.x) + st.session_state.padding,
                                                  st.session_state.grid_size)
            st.session_state.grid_y = np.linspace(min(st.session_state.y) - st.session_state.padding,
                                                  max(st.session_state.y) + st.session_state.padding,
                                                  st.session_state.grid_size)
            OK = OrdinaryKriging(
                st.session_state.x, st.session_state.y, st.session_state.z,
                variogram_model='gaussian',
                variogram_parameters={'sill': st.session_state.sill, 'range': st.session_state.range_,
                                      'nugget': st.session_state.nugget},
                nlags=10
            )
            st.session_state.z_pred, st.session_state.sigma = OK.execute('grid', st.session_state.grid_x,
                                                                         st.session_state.grid_y)
            st.success("Количество точек сброшено до исходного значения.")
            st.rerun()  # Используем st.rerun() для обновления интерфейса

        if target_points < total_points:
            # Точное вычисление шага для получения желаемого количества точек
            step_x = int(np.sqrt(total_points / target_points))
            step_y = int(np.sqrt(total_points / target_points))

            # Корректировка шага, чтобы количество точек было максимально близко к желаемому
            while (len(st.session_state.grid_x) // step_x) * (len(st.session_state.grid_y) // step_y) > target_points:
                step_x += 1
                step_y += 1

            # Уменьшение плотности точек
            st.session_state.grid_x = st.session_state.grid_x[::step_x]
            st.session_state.grid_y = st.session_state.grid_y[::step_y]
            st.session_state.z_pred = st.session_state.z_pred[::step_x, ::step_y]

            # Фактическое количество точек после уменьшения
            actual_points = len(st.session_state.grid_x) * len(st.session_state.grid_y)
            st.info(f"Фактическое количество точек после уменьшения: {actual_points}")

        # Сохранение в Excel
        results = pd.DataFrame({
            'X': np.repeat(st.session_state.grid_x, len(st.session_state.grid_y)),
            'Y': np.tile(st.session_state.grid_y, len(st.session_state.grid_x)),
            'Z_pred': st.session_state.z_pred.flatten()
        })
        st.write("Результаты кригинга:")
        st.write(results)

        # Кнопка для скачивания Excel
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            results.to_excel(writer, index=False)
        output_excel.seek(0)
        st.download_button(
            label="Скачать результаты в Excel",
            data=output_excel,
            file_name="kriging_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Сохранение изополей в DXF
        min_z_pred = np.min(st.session_state.z_pred)
        max_z_pred = np.max(st.session_state.z_pred)
        diff_z_pred = max_z_pred - min_z_pred
        st.write("Информация о Z_pred:")
        st.table(pd.DataFrame({
            "Параметр": ["Минимальное значение Z_pred", "Максимальное значение Z_pred", "Разница"],
            "Значение": [f"{min_z_pred:.2f}", f"{max_z_pred:.2f}", f"{diff_z_pred:.2f}"]
        }))

        step = st.number_input("Введите шаг горизонталей (например, 0.15 м):", value=0.15)
        if step <= 0:
            st.error("Шаг горизонталейй должен быть положительным числом.")
            return

        # Кнопка для формирования горизонталей
        if st.button("Сформировать горизонтали"):
            with st.spinner("Формирование горизонталей..."):
                st.session_state.dxf_file = create_dxf_file(st.session_state.grid_x, st.session_state.grid_y, st.session_state.z_pred, step)
                st.success("Горизонтали успешно сформированы!")

        # Кнопка для скачивания DXF-файла (активируется только после формирования горизонталей)
        if 'dxf_file' in st.session_state and st.session_state.dxf_file is not None:
            st.download_button(
                label="Скачать изополи в DXF",
                data=st.session_state.dxf_file,
                file_name="isolines.dxf",
                mime="application/dxf"
            )
        else:
            st.warning("Сначала сформируйте горизонтали.")

    except Exception as e:
        st.error(f"Не удалось сохранить результаты: {str(e)}")


# Основной интерфейс Streamlit
st.title("Кригинг с вариограммой Гаусса")
st.markdown("""
### Описание
Это приложение позволяет загружать данные, строить вариограммы и выполнять кригинг.
""")

# Вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Загрузка данных", "Вариограмма", "Редактирование вариограммы", "Кригинг", "Сохранение результатов"])

with tab1:
    st.header("Загрузка данных")
    load_data()

with tab2:
    st.header("Вариограмма")
    plot_empirical_variogram()

with tab3:
    st.header("Редактирование теоретической вариограммы")
    edit_variogram()

with tab4:
    st.header("Кригинг")
    run_kriging()

with tab5:
    st.header("Сохранение результатов")
    save_results()
