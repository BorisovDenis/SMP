import vtk
import numpy as np
import gmsh
import math
import os

# Класс расчётной сетки
class CalcMesh:

    # Конструктор сетки, полученной из stl-файла
    def __init__(self, nodes_coords, tetrs_points):
        # 3D-сетка из расчётных точек
        # Пройдём по узлам в модели gmsh и заберём из них координаты
        self.nodes = np.array([nodes_coords[0::3],nodes_coords[1::3],nodes_coords[2::3]])

        # Модельная скалярная величина распределена как-то вот так
        self.smth = np.power(self.nodes[0, :], 2) + np.power(self.nodes[1, :], 2)

        # Тут может быть скорость, но сейчас здесь нули
        self.velocity = 5 * np.ones(shape=(3, int(len(nodes_coords) / 3)), dtype=np.double)        

        length = max(self.nodes[2]) - min(self.nodes[2])
        min_coord = min(self.nodes[2])
        for i in range(int(len(nodes_coords) / 3)):            
            self.velocity[2][i] *= (length - self.nodes[2][i] + min_coord)/length
            self.velocity[1][i] *= (length - self.nodes[1][i] + min_coord)/length
            self.velocity[0][i] *= (length - self.nodes[0][i] + min_coord)/length


        self.time = 0

        # Пройдём по элементам в модели gmsh++
        self.tetrs = np.array([tetrs_points[0::4],tetrs_points[1::4],tetrs_points[2::4],tetrs_points[3::4]])
        self.tetrs -= 1

    # Метод отвечает за выполнение для всей сетки шага по времени величиной tau
    def move(self, tau):
        # По сути метод просто двигает все точки c их текущими скоростями
        self.time += tau
        self.nodes[2] += self.velocity[2] * np.sin(self.time/10 )
        self.nodes[1] += self.velocity[1] * np.sin(self.time )
        self.nodes[0] += self.velocity[0] * np.sin(self.time/10 )


    # Метод отвечает за запись текущего состояния сетки в снапшот в формате VTK
    def snapshot(self, snap_number):
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()

        # Скалярное поле на точках сетки
        smth = vtk.vtkDoubleArray()
        smth.SetName("smth")

        # Векторное поле на точках сетки
        vel = vtk.vtkDoubleArray()
        vel.SetNumberOfComponents(3)
        vel.SetName("vel")

        # Обходим все точки нашей расчётной сетки
        # Делаем это максимально неэффективным, зато наглядным образом
        for i in range(0, len(self.nodes[0])):
            # Вставляем новую точку в сетку VTK-снапшота
            points.InsertNextPoint(self.nodes[0,i], self.nodes[1,i], self.nodes[2,i])
            # Добавляем значение скалярного поля в этой точке
            smth.InsertNextValue(self.smth[i])
            # Добавляем значение векторного поля в этой точке
            vel.InsertNextTuple((self.velocity[0,i], self.velocity[1,i], self.velocity[2,i]))

        # Грузим точки в сетку
        unstructuredGrid.SetPoints(points)

        # Присоединяем векторное и скалярное поля к точкам
        unstructuredGrid.GetPointData().AddArray(smth)
        unstructuredGrid.GetPointData().AddArray(vel)

        # А теперь пишем, как наши точки объединены в тетраэдры
        # Делаем это максимально неэффективным, зато наглядным образом
        for i in range(0, len(self.tetrs[0])):
            tetr = vtk.vtkTetra()
            for j in range(0, 4):
                tetr.GetPointIds().SetId(j, self.tetrs[j,i])
            unstructuredGrid.InsertNextCell(tetr.GetCellType(), tetr.GetPointIds())

        # Создаём снапшот в файле с заданным именем
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(unstructuredGrid)
        writer.SetFileName("b/kill-" + str(snap_number) + ".vtu")
        writer.Write()


gmsh.initialize()

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'poor_cat.stl'))

gmsh.model.mesh.classifySurfaces(40 * math.pi / 180., True,
                                 False,
                                 180 * math.pi / 180.)

s = gmsh.model.getEntities(2)
l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)
'''
'''
# Теперь извлечём из gmsh данные об узлах сетки
nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()

# И данные об элементах сетки тоже извлечём, нам среди них нужны только тетраэдры, которыми залит объём
GMSH_TETR_CODE = 4
tetrsNodesTags = None
elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements()
for i in range(0, len(elementTypes)):
    if elementTypes[i] != GMSH_TETR_CODE:
        continue
    tetrsNodesTags = elementNodeTags[i]

if tetrsNodesTags is None:
    print("Can not find tetra data. Exiting.")
    gmsh.finalize()
    exit(-2)

print("The model has %d nodes and %d tetrs" % (len(nodeTags), len(tetrsNodesTags) / 4))

# На всякий случай проверим, что номера узлов идут подряд и без пробелов
for i in range(0, len(nodeTags)):
    # Индексация в gmsh начинается с 1, а не с нуля. Ну штош, значит так.
    assert (i == nodeTags[i] - 1)
# И ещё проверим, что в тетраэдрах что-то похожее на правду лежит.
assert(len(tetrsNodesTags) % 4 == 0)

# TODO: неплохо бы полноценно данные сетки проверять, да
size = 10
mesh = CalcMesh(nodesCoord, tetrsNodesTags)
mesh.snapshot(0)

tau = 0.00001

for i in range(1, 500):
    mesh.move(tau)
    mesh.snapshot(i)
'''
'''
gmsh.finalize()

#gmsh.model.mesh.createGeometry()
