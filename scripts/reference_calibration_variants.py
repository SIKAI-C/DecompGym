"""Prospective reference variants, executed only inside the calibration bundle."""


def run_budgeted_benders():
    solver = ReferenceClass(verbose=False)

    def saved_incumbent():
        incumbent = solver._incumbent
        if incumbent is None or not math.isfinite(solver.global_ub):
            return None
        return {"best_objective": solver.global_ub, "global_lb": solver.global_lb, "global_ub": solver.global_ub, "final_vars": incumbent["vars"], "master_metadata": incumbent["master_metadata"], "sub_metadata": incumbent["sub_metadata"], "iters": len(solver.iter_records), "total_time": time.monotonic() - STARTED}

    original_termination = solver.should_terminate

    def terminate(iteration, master, subproblem, lower, upper):
        if subproblem.all_feasible and math.isfinite(lower) and math.isfinite(upper) and upper >= lower - 1e-6:
            if (upper - lower) / max(1.0, abs(upper)) <= 0.005:
                return True
        return original_termination(iteration, master, subproblem, lower, upper)

    def checkpoint(*_args):
        result = saved_incumbent()
        if result:
            publish(result, 'incumbent_checkpoint')

    solver.should_terminate = terminate
    solver.on_cuts_added = checkpoint
    try:
        publish(solver.run('../../data.json'), 'decomposition_stop_with_half_percent_gap_target')
    except CalibrationDeadline:
        result = saved_incumbent()
        if result is None:
            raise
        publish(result, '54_second_internal_deadline_with_saved_incumbent')


def run_pool_bin_packing():
    from src.cg.bp.a_gold_cg import _aggregate_items
    from src.utils.base_cg_class import PricingResult

    class PoolColumns(ReferenceClass):
        def initial_patterns(self, inst):
            sizes, demands = _aggregate_items(inst)
            patterns = []
            for index, size in enumerate(sizes):
                pattern = [0] * len(sizes)
                pattern[index] = min(demands[index], inst.bin_capacity // size)
                if pattern[index]:
                    patterns.append(pattern)
            return patterns

        def solve_pricing(self, inst, master_solution, patterns):
            sizes, demands = _aggregate_items(inst)
            model = gp.Model('bp_knapsack_pricing_pool')
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = 64
            model.Params.MIPGap = 0
            variables = model.addVars(len(sizes), vtype=GRB.INTEGER, lb=0, ub={i: demands[i] for i in range(len(sizes))})
            model.addConstr(gp.quicksum(sizes[i] * variables[i] for i in range(len(sizes))) <= inst.bin_capacity)
            model.addConstr(gp.quicksum(variables[i] for i in range(len(sizes))) >= 1)
            model.setObjective(gp.quicksum(master_solution.duals[i] * variables[i] for i in range(len(sizes))), GRB.MAXIMIZE)
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError('Pricing pool did not finish within the remaining budget')
            generated = []
            for index in range(model.SolCount):
                model.Params.SolutionNumber = index
                generated.append([round(variables[i].Xn) for i in range(len(sizes))])
            best = generated[0]
            reduced_cost = 1 - sum(master_solution.duals[i] * best[i] for i in range(len(sizes)))
            known = {tuple(pattern) for pattern in patterns}
            for pattern in generated:
                if pattern == best and reduced_cost < -self.rc_tolerance:
                    continue
                if tuple(pattern) not in known:
                    patterns.append(pattern)
                    known.add(tuple(pattern))
            return PricingResult(pattern=best, reduced_cost=reduced_cost, metadata={"pricing_pool_solutions": len(generated)})

    publish(PoolColumns(verbose=False).run('../../data.json'), 'column_generation_with_pricing_pool_and_integer_restricted_master')


def run_phase_one_gap():
    from src.cg.gap.a_gold_cg import _extract
    solver = ReferenceClass(verbose=False)
    inst = solver.load_instance('../../data.json')
    solver._init_matrices(inst)
    weights, costs, capacities = _extract(inst)
    jobs, machines = len(weights), len(capacities)
    allowed = {(job.id, assignment.machine_id) for job in inst.jobs for assignment in job.assignments}
    patterns = []
    for machine in range(machines):
        patterns.append({"machine": machine, "jobs": [0] * jobs})
        for job in range(jobs):
            if (job, machine) in allowed and weights[job][machine] <= capacities[machine]:
                vector = [0] * jobs
                vector[job] = 1
                patterns.append({"machine": machine, "jobs": vector})
    known = {(pattern['machine'], tuple(pattern['jobs'])) for pattern in patterns}
    phase_one, lp_bound, iterations = True, None, 0
    for iterations in range(1, 201):
        master = gp.Model('gap_phase1_restricted_master' if phase_one else 'gap_restricted_master')
        lam = master.addVars(len(patterns), lb=0)
        artificial = master.addVars(jobs, lb=0) if phase_one else None
        covers = [master.addConstr(gp.quicksum(pattern['jobs'][job] * lam[index] for index, pattern in enumerate(patterns)) + (artificial[job] if phase_one else 0) == 1) for job in range(jobs)]
        convexity = [master.addConstr(gp.quicksum(lam[index] for index, pattern in enumerate(patterns) if pattern['machine'] == machine) == 1) for machine in range(machines)]
        objective = gp.quicksum(artificial[job] for job in range(jobs)) if phase_one else gp.quicksum(sum(costs[job][pattern['machine']] * pattern['jobs'][job] for job in range(jobs)) * lam[index] for index, pattern in enumerate(patterns))
        master.setObjective(objective, GRB.MINIMIZE)
        master.optimize()
        if master.Status != GRB.OPTIMAL:
            raise RuntimeError('Restricted master not optimal')
        if phase_one and master.ObjVal <= 1e-7:
            phase_one = False
            continue
        dual_jobs = [constraint.Pi for constraint in covers]
        dual_machines = [constraint.Pi for constraint in convexity]
        improving = False
        for machine in range(machines):
            pricing = gp.Model('gap_machine_knapsack_pricing')
            pricing.Params.PoolSearchMode = 2
            pricing.Params.PoolSolutions = 64
            pricing.Params.MIPGap = 0
            z = pricing.addVars(jobs, vtype=GRB.BINARY, ub={job: int((job, machine) in allowed) for job in range(jobs)})
            pricing.addConstr(gp.quicksum(weights[job][machine] * z[job] for job in range(jobs)) <= capacities[machine])
            pricing.setObjective(gp.quicksum(((0 if phase_one else costs[job][machine]) - dual_jobs[job]) * z[job] for job in range(jobs)) - dual_machines[machine], GRB.MINIMIZE)
            pricing.optimize()
            if pricing.Status != GRB.OPTIMAL:
                raise RuntimeError('Pricing not optimal')
            improving = improving or pricing.ObjVal < -1e-7
            for index in range(pricing.SolCount):
                pricing.Params.SolutionNumber = index
                vector = [round(z[job].Xn) for job in range(jobs)]
                key = (machine, tuple(vector))
                if key not in known:
                    patterns.append({"machine": machine, "jobs": vector})
                    known.add(key)
        if not improving:
            if phase_one:
                raise RuntimeError('Phase-I pricing could not remove all artificial variables')
            lp_bound = float(master.ObjVal)
            break
    integer = solver.solve_integer_master(inst, patterns)
    result = {"best_objective": integer.objective, "global_lb": lp_bound, "global_ub": integer.objective, "iters": iterations, "total_time": time.monotonic() - STARTED, "master_metadata": integer.metadata or {}, "sub_metadata": {}}
    publish(result, 'phase1_column_generation_with_pricing_pool_and_integer_restricted_master')


def run_variant():
    if SUITE == 'bd':
        run_budgeted_benders()
    elif PROBLEM == 'bp':
        run_pool_bin_packing()
    elif PROBLEM == 'gap':
        run_phase_one_gap()
    else:
        publish(ReferenceClass(verbose=False).run('../../data.json'))
