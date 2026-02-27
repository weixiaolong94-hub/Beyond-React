"""
Data preprocessing
"""
import argparse
import json
import os
from evaluation import ExecutionGraph,ExecutionNode
import random
random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument('--answer_dir',type=str, required=True,help='where the answers stored.')
parser.add_argument('--method',type=str,required=True,help='the name of the method.')
parser.add_argument('--output', type=str, default="converted_answers.json", required=False, help='output path for the converted answer.')


def generate_init_message_node(eg:ExecutionGraph,functions,query):
    init_node = ExecutionNode(role='system', message="You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say \"I give up and restart\".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet's Begin!\nTask description: You should use functions to help handle the real time user querys. Remember to ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user.\nSpecifically, you have access to the following functions: " + str(functions))
    eg.set_init_node(init_node)
    
    node = ExecutionNode(role='user', message=query)
    eg.add_node(node)
    eg[init_node,node] = None
    return node



# def process_valid_data(method,answer_generation):
#     conversation = answer_generation['train_messages'][-1]
#     functions = answer_generation['function']
#     query = answer_generation['query']
#     eg = ExecutionGraph()
#     last_node = generate_init_message_node(eg,functions,query)
    
#     index = 2
#     while index < len(conversation):
#         message = conversation[index]
#         role = message['role']
#         if role == 'system' or role == 'user' or role == 'function' or role == 'tool':
#             index = index + 1
#             continue
#         elif role == 'assistant':
#             if 'function_call' in message :
#                 node = ExecutionNode(role='tool', message={
#                     'name':message['function_call']['name'],
#                     'arguments':message['function_call']['arguments'],
#                     'response':conversation[index+1]['content'] if message['function_call']['name']!='Finish' else ''
#                     })
#                 index = index + 1
#             elif 'tool_calls' in message and message['tool_calls'] is not None:
#                 calls = message['tool_calls']
#                 for tc in calls:
#                     id, function = tc['id'], tc['function']
#                     name, arguments = function['name'], function['arguments']
#                     if name == 'Finish':
#                         node = ExecutionNode(role='tool', message={
#                             'name':name,
#                             'arguments':arguments,
#                             'response':''
#                         })
#                         break
#                     else:
#                         for message2 in conversation[index+1:]:
#                             if message2['role'] == 'tool'  and message2['tool_call_id'] == id:
#                                 response = message2['content']
#                                 node = ExecutionNode(role='tool', message={
#                                     'name':name,
#                                     'arguments':arguments,
#                                     'response':response
#                                     })
#                                 eg.add_node(node)
#                                 eg[last_node,node] = None
#                                 last_node = node
#                                 break
#             else:
#                 node = ExecutionNode(role='assistant',
#                                         message=message['content'])
            
                
#         else:
#             raise NotImplementedError(f'Unkown role {role}')
        
#         index = index + 1
#         if last_node != node:
#             eg.add_node(node)
#             eg[last_node,node] = None
#             last_node = node
    
#     eg = eg.reduce_graph_to_sequence()
    
#     return {
#         'query':query,
#         'available_tools':functions,
#         'answer':{
#             'method':method,
#             'total_steps': eg.node_count,
#             'final_answer': answer_generation['final_answer'],
#             'answer_details': eg.convert_to_dict()
#         }
#     }
# In convert_to_answer_format.py

def process_valid_data(method, answer_generation):
    # train_messages 可能不存在或为空，使用 .get() 更安全
    conversation = answer_generation.get('train_messages', [])
    # train_messages 是一个列表的列表，通常我们取最后一个
    if conversation:
        conversation = conversation[-1]
    
    functions = answer_generation.get('function', [])
    query = answer_generation.get('query', '')
    eg = ExecutionGraph()
    last_node = generate_init_message_node(eg, functions, query)
    
    index = 0
    while index < len(conversation):
        message = conversation[index]
        role = message.get('role')

        # 如果角色是输入或观察，直接跳过
        if role in ['system', 'user', 'function', 'tool']:
            index += 1
            continue
        
        elif role == 'assistant':
            node = None  # 在循环开始时将 node 初始化为 None
            
            # 优先处理工具调用
            if 'tool_calls' in message and message.get('tool_calls'):
                # ... (您的 tool_calls 处理逻辑) ...
                # 为简化，我们只处理第一个调用
                tc = message['tool_calls'][0]
                tool_id, function = tc.get('id'), tc.get('function', {})
                name, arguments = function.get('name'), function.get('arguments')

                response = f"Error: Corresponding tool response not found for tool_call_id {tool_id}."
                # 向后查找对应的tool消息
                for next_idx in range(index + 1, len(conversation)):
                    next_message = conversation[next_idx]
                    if next_message.get('role') == 'tool' and next_message.get('tool_call_id') == tool_id:
                        response = next_message.get('content', '')
                        break
                
                node = ExecutionNode(role='tool', message={
                    'name': name,
                    'arguments': arguments,
                    'response': response
                })

            # 兼容旧的 function_call 格式
            elif 'function_call' in message and message.get('function_call'):
                fc = message['function_call']
                name, arguments = fc.get('name'), fc.get('arguments')
                response = "Error: Corresponding function response not found."
                if index + 1 < len(conversation) and conversation[index+1].get('role') in ['function', 'tool']:
                    response = conversation[index+1].get('content', '')

                node = ExecutionNode(role='tool', message={
                    'name': name,
                    'arguments': arguments,
                    'response': response
                })

            # 如果没有工具调用，则认为是思考
            elif message.get('content'):
                node = ExecutionNode(role='assistant', message=message['content'])
            
            # ==================== 这是关键的修复 ====================
            # 只有当一个新节点被成功创建时，才进行后续操作
            if node is not None:
                if last_node != node: # 现在 node 肯定有值
                    eg.add_node(node)
                    eg[last_node, node] = None
                    last_node = node
            else:
                # 如果这是一个空的 assistant 消息，我们就打印一个警告并忽略它
                print(f"Warning: Skipping an empty assistant message at index {index}.")
            # =======================================================
                
        else:
            print(f'Warning: Unknown role found in valid data processing: {role}')
        
        index += 1
    
    eg = eg.reduce_graph_to_sequence()
    
    return {
        'query': query,
        'available_tools': functions,
        'answer': {
            'method': method,
            'total_steps': eg.node_count,
            'final_answer': answer_generation.get('final_answer', ''),
            'answer_details': eg.convert_to_dict()
        }
    }
def process_invalid_data(method,data_dict):
    answer_generation = data_dict['answer_generation']
    functions = answer_generation['function']
    query = answer_generation['query']
    eg = ExecutionGraph()
    last_node = generate_init_message_node(eg,functions,query)
    if 'CoT' in method or 'cot' in method:
        trail = random.choice(data_dict["trys"])

        
        index = 0
        while index < len(trail['chain']):
            message = trail['chain'][index]
            if message['node_type'] == 'Action':
                node = ExecutionNode(role='tool', message={
                    'name':message['description'],
                    'arguments':(trail['chain'][index+1]['description']),
                    'response':(trail['chain'][index+1]['observation'])})
            
                index = index + 1
            elif message['node_type'] == 'Thought':
                node = ExecutionNode(role='assistant',
                                        message=message['description'])
            else:
                raise NotImplementedError(f"Unknown node_type: {message['node_type']}")
            index = index + 1

            eg.add_node(node)
            eg[last_node,node] = None
            last_node = node
        eg = eg.reduce_graph_to_sequence()

    elif 'DAG' in method:
        # DAG 日志的执行链直接在 'execution_chain' 字段中
        if 'execution_chain' not in data_dict or not data_dict['execution_chain']:
            print(f"Warning: 'execution_chain' not found or is empty for DAG method in file. Skipping.")
            # 返回一个只包含初始节点的空图
            return {
                'query': query,
                'available_tools': functions,
                'answer': {
                    'method': method,
                    'total_steps': eg.node_count,
                    'final_answer': "Planning or execution failed before any steps were taken.",
                    'answer_details': eg.convert_to_dict()
                }
            }
        
        execution_chain = data_dict['execution_chain']
        
        index = 0
        while index < len(execution_chain):
            message = execution_chain[index]
            # 逻辑与CoT分支完全相同
            if message['node_type'] == 'Action':
                # 检查是否存在下一个节点，防止索引越界
                if index + 1 < len(execution_chain):
                    next_message = execution_chain[index+1]
                    node = ExecutionNode(role='tool', message={
                        'name': message['description'],
                        'arguments': next_message.get('description', ''), # 使用.get增加稳健性
                        'response': next_message.get('observation', '')
                    })
                    index = index + 1 # 因为合并了两个节点，所以索引要多加1
                else:
                    # 如果Action节点是最后一个，则认为它是一个未完成的调用
                    node = ExecutionNode(role='tool', message={
                        'name': message['description'],
                        'arguments': '{}',
                        'response': '{"error": "Execution stopped after action selection."}'
                    })

            elif message['node_type'] == 'Action Input':
                # 如果我们直接遇到了一个Action Input，说明它前面没有Action节点
                # 这种情况不应该发生，但为了稳健性，我们还是处理一下
                print(f"Warning: Found an orphaned 'Action Input' node at index {index}.")
                node = ExecutionNode(role='tool', message={
                    'name': 'unknown_action',
                    'arguments': message.get('description', ''),
                    'response': message.get('observation', '')
                })

            elif message['node_type'] == 'Thought':
                node = ExecutionNode(role='assistant', message=message['description'])
            
            else:
                # 忽略其他类型的节点，如初始节点，因为我们已经手动创建了
                print(f"Ignoring node with unhandled type: {message.get('node_type', 'N/A')}")
                index = index + 1
                continue # 跳过后续的节点添加逻辑

            index = index + 1

            eg.add_node(node)
            eg[last_node, node] = None
            last_node = node
        eg = eg.reduce_graph_to_sequence()
   
    elif 'DFS' in method or 'dfs' in method:

        def DFS(root):
            if len(root['children']) == 0:
                node = ExecutionNode(role=root['node_type'],message=root)
                eg.add_node(node)
                return node
            else:
                child_nodes = [DFS(node) for node in root['children']]
                root['children'] = None
                root_node = ExecutionNode(role=root['node_type'],message=root)
                eg.add_node(root_node)
                for child_node in child_nodes:
                    eg.add_edge(root_node,child_node)
                return root_node
        for node in data_dict['tree']['tree']['children']:
            eg[last_node,DFS(node)] = None

        
        # purify the graph
        def purify_graph(node:ExecutionNode):
            if node.role == 'Action':
                adj_nodes = eg.get_adjacent_node(node)
                for adj_node in adj_nodes:
                    adj_node = eg[adj_node]
                    if adj_node.role == 'Action Input':
                        node.role = 'tool'
                        node.message = {
                            'name':node.message['description'],
                            'arguments':(adj_node.message['description']),
                            'response':(adj_node.message['observation'])
                            
                        }
                        # remove adj_node
                        adj_node = eg.pop_node(adj_node)
                        to_nodes = eg.edges.pop(adj_node.node_id,{})
                        eg.edges[node.node_id].update(to_nodes)
                        eg.edges[node.node_id].pop(adj_node.node_id)
                        node.out_degree += len(to_nodes)
                        break
            elif node.role == 'Thought':
                node.role = 'assistant'
                node.message = node.message['description']
            elif node.role == 'Action Input':
                print('Founding Extra Action Input Node')
                pass
            elif node.role =='system' or node.role=='user':
                pass
            else:
                raise Exception('Unknown role {}'.format(node.role))
            adj_nodes = eg.get_adjacent_node(node)
            for adj_node in adj_nodes:
                purify_graph(eg[adj_node])
            
        purify_graph(last_node)
        eg = eg.reduce_graph_to_sequence()
    else:
        raise NotImplementedError(f'Unknown method {method}')
    return {
        'query':query,
        'available_tools':functions,
        'answer':{
            'method':method,
            'total_steps': eg.node_count,
            'final_answer': answer_generation['final_answer'],
            'answer_details': eg.convert_to_dict()
        }
    }
             
                    
                    
if __name__=='__main__':
    args = parser.parse_args()
    answer_dir = args.answer_dir
    method = args.method
    output = args.output
    answer_dict = {}
    for filename in os.listdir(answer_dir):
        if filename.endswith('.json') and method in filename:
            qid = filename.split('_')[0]
            data_dict = json.load(open(os.path.join(answer_dir,filename)))
            if not data_dict['answer_generation']['valid_data']:
                answer_dict[qid] = process_invalid_data(method,data_dict)
            else:
                answer_dict[qid] = process_valid_data(method,data_dict['answer_generation'])
    print(f'Converted {len(answer_dict)} answers')
    json.dump(answer_dict,open(output,'w'), indent=2)