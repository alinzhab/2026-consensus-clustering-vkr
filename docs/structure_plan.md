# План реструктуризации проекта (для следующей итерации)

Документ фиксирует целевую структуру проекта и обоснование, почему
полная перекладка не выполнялась прямо сейчас, к моменту защиты ВКР.

## Что уже сделано

- Каталог ядра переименован: `python_port/` → `consensus_lab/`. Имя
  `python_port` вводило в заблуждение и создавало вопрос об авторстве
  кода (см. `docs/limitations.md`, раздел 6). Новое имя отражает
  назначение каталога — исследовательская «лаборатория» алгоритмов
  консенсус-кластеризации.
- Удалён неиспользуемый `python_port/benchmark.py` — он дублировал
  функциональность `run_all_benchmarks.py` и не импортировался ни из
  `app.py`, ни из тестов, ни из `experiments/`.
- Обновлены все ссылки в `app.py`, `tests/conftest.py`, `experiments/*`,
  `README.md`, `ANALYSIS_RUNBOOK.md`, `docs/*`, `run_sdgca_linkage_suite.bat`.
- Из `run_sdgca_linkage_suite.bat` убран хардкод
  `C:\Users\tikho\anaconda3\python.exe`. Скрипт теперь сам выбирает
  `.venv\Scripts\python.exe`, если виртуальное окружение есть, иначе
  использует `python` из `PATH`.
- В `app.py` единый интерфейс алгоритмов (`AlgorithmRegistry`) уже
  присутствует и используется. Дублирование между парами baseline/modified
  на уровне CLI помечено как технический долг (см. ниже).

## Целевая структура

Полная иерархия, к которой проект движется:

```text
2026_Consensus_Clustering/
  src/consensus/                  # ядро как импортируемый пакет
    algorithms/
      base.py                     # абстрактный интерфейс ConsensusAlgorithm
      registry.py                 # реестр алгоритмов
      hierarchical/baseline.py
      hierarchical/weighted.py
      sdgca/baseline.py
      sdgca/modified.py
    data/
      loaders.py                  # load_dataset_full + валидация
      base_clusterings.py
      generators/{densired,repliclust,simple}.py
    evaluation/
      metrics.py
      statistical_tests.py        # Фридман, Nemenyi, Wilcoxon-Holm, bootstrap CI
    experiments/
      runner.py                   # один общий runner вместо четырёх дублей CLI
      analysis_suite.py
      reports.py                  # сборка markdown
    web/
      __init__.py                 # create_app()
      blueprints/{datasets,experiments,results,generate}.py
      db.py
  tests/
    test_metrics.py
    test_consensus_smoke.py
    test_loaders.py
  datasets/
  results/
    raw/                          # JSON по запускам
    summary/                      # TSV сводки
    reports/                      # analysis_report.md
  docs/
    thesis_outline.md
    contribution.md
    limitations.md
    structure_plan.md
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  .github/workflows/ci.yml
  Makefile
  README.md
  README.en.md
```

Каждое изменение защищает на конкретный вопрос комиссии:

| Изменение                            | Какой вопрос комиссии закрывает                                  |
|--------------------------------------|------------------------------------------------------------------|
| `src/consensus/` как пакет           | «Это ad-hoc набор скриптов или полноценная библиотека?»          |
| `algorithms/{hierarchical,sdgca}/`   | «Как организован код? Можно ли добавить пятый алгоритм?»         |
| `evaluation/`                        | «Где код метрик и статистических тестов? Они независимы?»        |
| `experiments/runner.py` (один)       | «Почему 4 почти одинаковых CLI? Не дублирование?»                |
| `web/blueprints/`                    | «`app.py` 1100 строк — это монолит?»                             |
| `results/{raw,summary,reports}/`     | «Сырые результаты и отчёты лежат вперемешку — почему?»           |
| `pyproject.toml`                     | «Как собрать? Современный ли инструментарий?»                    |
| `.github/workflows/ci.yml`           | «Тесты вообще запускаются? Где доказательство, что они зелёные?» |
| `README.en.md`                       | «Если работа имеет международную ценность, где английская версия?» |

## Почему перекладка не выполнена прямо сейчас

Сознательное решение, принятое за неделю до защиты:

1. **Высокий риск к защите**. Перенос `consensus_lab/*` в `src/consensus/...`
   с превращением каталога в импортируемый пакет (`from
   consensus.algorithms.sdgca import baseline`) затрагивает: внутренние
   импорты алгоритмов друг на друга (`sdgca_modified` импортирует из
   `sdgca`), `app.py`, все `experiments/`, все `tests/`, все `.bat` и
   команды в README. Любой пропущенный путь → сломанная демонстрация
   приложения на защите.
2. **Отсутствие времени на ручное тестирование всех путей**. Полный
   `run_full_analysis_suite.py` в режиме `--runs 5 --m 20` идёт часами;
   быстрая проверка только `--runs 1 --m 10 --methods average` не
   покрывает все сценарии (например, `ward` linkage, или режим
   `--resume`).
3. **Малая выгода на этапе предзащиты**. Текущая плоская структура
   `consensus_lab/` уже понятна рецензенту: один каталог, одни тесты,
   одно приложение. Ключевые претензии (имя `python_port`, мёртвые
   файлы, хардкод путей в `.bat`) закрыты минимальным рефакторингом.

## Дальнейшие шаги (после защиты)

1. **Сделать `consensus_lab/` импортируемым пакетом**. Добавить
   `__init__.py`, переписать импорты с явных `from sdgca import ...` на
   `from consensus_lab.sdgca import ...`. Заодно убрать
   `sys.path.insert(...)` из `app.py`, тестов и экспериментов.
2. **Объединить четыре CLI в один `runner.py`**. Каждая `run_*` функция
   — это просто адаптер над `AlgorithmRegistry`, его уже достаточно;
   нужен единый CLI, принимающий `--algorithm`, который и выбирает
   реализацию из реестра. Это устраняет дублирование `argparse`-обвязки
   на 200+ строк суммарно.
3. **Разделить `app.py` на blueprints**. `web/blueprints/datasets.py`,
   `web/blueprints/experiments.py`, `web/blueprints/results.py`,
   `web/blueprints/generate.py`. Создать `create_app()` фабрику. Это
   стандартная Flask-практика; в текущем виде 1119 строк в одном файле
   технически работают, но для рецензента — слабое место.
4. **Разделить `results/` на `raw/`, `summary/`, `reports/`**. Сейчас
   там вперемешку JSON по запускам, TSV сводки и Markdown-отчёты.
   Разделение делает структуру каталога самоописательной.
5. **Добавить `pyproject.toml`** (PEP 621). Перенести метаданные
   проекта, опциональные зависимости (`[project.optional-dependencies]
   dev = [...]`). Оставить `requirements.txt` как pin-файл для
   `render.yaml`.
6. **Добавить `.github/workflows/ci.yml`**. Минимум: `pytest tests/` на
   каждый push. Это самый дешёвый способ показать рецензенту, что
   тесты не декоративные.
7. **Английский `README.en.md`**. Если работа выходит за пределы
   локальной комиссии — обязательно. Краткая версия `README.md` плюс
   «Run examples» — достаточно.

## Принцип, которым стоит руководствоваться

Рефакторинг ради рефакторинга вреден перед защитой. Любая структурная
перекладка должна:

- иметь явную цель (какой вопрос комиссии она закрывает);
- быть полностью покрыта тестами или ручными прогонами «до/после»;
- не блокировать запуск `app.py` и `pytest tests/`.

После защиты можно идти к целевой структуре итеративно, по одному
пункту за PR.
