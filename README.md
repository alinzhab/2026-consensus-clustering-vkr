# 2026_Consensus_Clustering

Этот репозиторий собран для экспериментов с consensus clustering, иерархической агрегацией и SDGCA на реальных и синтетических датасетах.

## Структура проекта

- `datasets/` — датасеты в форматах `.mat` и `.npz`
- `python_port/hierarchical_consensus.py` — базовая версия hierarchical consensus clustering
- `python_port/hierarchical_consensus_modified.py` — модифицированная weighted-версия hierarchical consensus
- `python_port/sdgca.py` — Python-версия SDGCA с поддержкой разных `linkage`
- `python_port/sdgca_modified.py` — модифицированная версия SDGCA с отдельными параметрами
- `python_port/synthetic_data.py` — простой генератор синтетических датасетов
- `python_port/densired_style_generator.py` — генератор сложных синтетических данных в стиле DENSIRED
- `python_port/benchmark.py` — запуск бенчмарков
- `results/` — результаты запусков и таблицы с метриками

## Какие данные есть

В папке `datasets/` лежат:

- реальные `.mat` датасеты
- простые синтетические `.npz` датасеты:
  - `synthetic_easy`
  - `synthetic_overlap`
  - `synthetic_hard`
- сложные синтетические `.npz` датасеты в стиле DENSIRED:
  - `densired_compact_hard`
  - `densired_stretched_hard`
  - `densired_mix_hard`

Формат `.npz` датасетов:

- `X` — матрица признаков объектов
- `gt` — истинные метки классов
- `members` — набор базовых кластеризаций для consensus clustering
- `meta` — параметры генерации

## Быстрый старт

Перейти в папку с Python-кодом:

```bash
cd python_port
```

Сгенерировать простые синтетические данные:

```bash
C:\Users\tikho\anaconda3\python.exe synthetic_data.py
```

Сгенерировать сложные DENSIRED-style датасеты:

```bash
C:\Users\tikho\anaconda3\python.exe densired_style_generator.py --preset all
```

Запустить базовый hierarchical consensus:

```bash
C:\Users\tikho\anaconda3\python.exe hierarchical_consensus.py --dataset synthetic_easy --root ..\datasets
```

Запустить модифицированный hierarchical consensus:

```bash
C:\Users\tikho\anaconda3\python.exe hierarchical_consensus_modified.py --dataset densired_mix_hard --root ..\datasets --sharpen 1.5
```

Запустить базовый SDGCA:

```bash
C:\Users\tikho\anaconda3\python.exe sdgca.py --dataset densired_compact_hard --root ..\datasets --method complete
```

Запустить модифицированный SDGCA:

```bash
C:\Users\tikho\anaconda3\python.exe sdgca_modified.py --dataset densired_compact_hard --root ..\datasets --method complete
```

## Основные параметры

Для `hierarchical_consensus.py` и `hierarchical_consensus_modified.py`:

- `--dataset` — имя датасета без расширения
- `--root` — путь к папке `datasets`
- `--m` — сколько базовых кластеризаций брать
- `--runs` — сколько повторов делать
- `--method` — тип иерархической агрегации: `average`, `complete`, `single`

Дополнительно для `hierarchical_consensus_modified.py`:

- `--sharpen` — усиление весов базовых кластеризаций

Для `sdgca.py`:

- `--dataset`
- `--root`
- `--m`
- `--runs`
- `--lambda_`
- `--eta`
- `--theta`
- `--method`

Для `sdgca_modified.py`:

- используются отдельные параметры по умолчанию для разных датасетов
- при необходимости их можно переопределить через `--lambda_`, `--eta`, `--theta`

## Как устроены синтетические генераторы

### `synthetic_data.py`

Это простой генератор.

Он:

- создает истинные метки `gt`
- вносит шум в метки
- строит матрицу `members`

Этот вариант полезен для быстрой проверки пайплайна, но он не генерирует сложную геометрию точек.

### `densired_style_generator.py`

Это более сложный генератор.

Он:

- строит скелет кластера из множества `core`-точек
- использует random walk
- поддерживает ветвление
- задает разную плотность кластеров
- добавляет шум
- строит и сами точки `X`, и `members`

Такой генератор лучше подходит для трудных тестов consensus clustering.

## Метрики качества

Во всех основных скриптах считаются:

- `NMI`
- `ARI`
- `F-score`

Обычно:

- чем ближе значение к `1`, тем лучше
- дополнительно выводится разброс по нескольким повторам

## Бенчмарки

Пример запуска бенчмарка для hierarchical consensus:

```bash
C:\Users\tikho\anaconda3\python.exe benchmark.py --mode baseline --dataset Ecoli Lung --root ..\datasets
```

Результаты сохраняются в папку `results/`.

## Что уже проверялось

В проекте уже тестировались:

- baseline vs weighted hierarchical consensus
- baseline SDGCA vs modified SDGCA
- разные `linkage`-стратегии: `average`, `complete`, `single`
- сложные DENSIRED-style датасеты

На `densired_compact_hard` лучший результат среди проверенных вариантов показал `modified SDGCA + complete linkage`.

## Замечание по окружению

Для запусков использовался:

```bash
C:\Users\tikho\anaconda3\python.exe
```

Если запускать другим интерпретатором, убедись, что установлены:

```bash
numpy
scipy
```
