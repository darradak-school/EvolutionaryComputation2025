import ioh
import importlib.metadata
print(importlib.metadata.version("ioh"))

from ioh import get_problem, ProblemClass, logger

# Create a problem
p = get_problem("OneMax", 1, 10, ProblemClass.PBO)

# Attach a logger
log = logger.Analyzer(root=".", folder_name="demo_logs", algorithm_name="demo")
p.attach_logger(log)

# Evaluate
print("f([1]*10) =", p([1]*10))

log.close()  # in some versions it's .close() not .finish_logging()