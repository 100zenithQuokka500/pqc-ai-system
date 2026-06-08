@app.get('/policy/history')
async def policy_history():
    return {
        'rewards': policy_engine.reward_history,
        'epsilons': policy_engine.epsilon_history,
        'decisions': policy_engine.decision_log
    }