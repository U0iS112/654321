from blackboard.evaluation_methods.methods import candidate_test, post_processing_stuff , final_evalutions_signals

def run_evaluations(export_dir = ""):
    if export_dir:
        candidate_test.run(export_dir)
        post_processing_stuff.run(export_dir)
        final_evalutions_signals.run(export_dir)