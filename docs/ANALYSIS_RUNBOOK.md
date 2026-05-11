# Запуск подробного анализа алгоритмов

Этот файл описывает полный исследовательский сценарий:

```text
сгенерировать большие датасеты разных типов
-> построить базовые кластеризации members
-> прогнать 4 consensus-алгоритма по 4 linkage
-> получить итоговый Markdown-отчёт
```

## 1. Подготовить окружение

```powershell
cd "C:\Users\tikho\Downloads\Telegram Desktop\2026_Consensus_Clustering-"
.\.venv\Scripts\Activate.ps1
```

Если `.venv` ещё нет:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Сгенерировать большие датасеты для анализа

```powershell
python consensus_lab\generate_analysis_datasets.py --base-clusterings 40
```

Будут созданы датасеты с префиксом `analysis_`:

- обычные хорошо разделённые кластеры
- обычные перекрывающиеся кластеры
- высокоразмерные данные
- дисбаланс классов
- плотностные DENSIRED-like данные
- вытянутые/сложные кластеры
- archetype/repliclust-like данные

## 3. Быстрая проверка перед долгим запуском

```powershell
python consensus_lab\run_full_analysis_suite.py --runs 1 --m 10 --methods average --algorithms hierarchical_baseline hierarchical_weighted
```

Если эта команда работает, можно запускать полный анализ.

## 4. Полный долгий анализ

```powershell
python consensus_lab\run_full_analysis_suite.py --runs 3 --m 20
```

Скрипт запускает:

- `hierarchical_baseline`
- `hierarchical_weighted`
- `sdgca`
- `sdgca_modified`

Для каждого linkage:

- `average`
- `complete`
- `single`
- `ward`

Результаты пишутся построчно в:

```text
results/analysis_full_suite.tsv
```

Если остановить процесс, повторный запуск продолжит с места остановки и пропустит уже готовые строки.

## 5. Отдельный очень долгий режим

Для более устойчивой статистики:

```powershell
python consensus_lab\run_full_analysis_suite.py --runs 5 --m 20
```

Этот режим может считаться много часов, потому что SDGCA строит несколько матриц размера `n x n`.

## 6. Построить итоговый отчёт

```powershell
python consensus_lab\analyze_benchmark_results.py
```

Отчёт появится здесь:

```text
results/analysis_report.md
```

## Важно

Не начинайте сразу с 10000 объектов. Текущие алгоритмы строят матрицы `n x n`, поэтому память растёт квадратично.

Практичный размер для долгого анализа на локальном компьютере:

```text
3000-4000 объектов
30-40 базовых кластеризаций
runs = 3 или 5
m = 20
```
