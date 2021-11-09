from kazoo.client import KazooClient
import configparser

import mysql.connector
from mysql.connector import errorcode

from ecosys.utils.database import DatabaseHelper

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('coordinator.cfg')

db_helper = DatabaseHelper(
    host=config['mysql']['host'], 
    host=config['mysql']['user'], 
)

# class Coordinator(CoordinatorManagementServicer):
#     def __init__(self):
#         super(Coordinator, self).__init__()
#         self.cnx = mysql.connector.connect(
#             host="mysql.default.svc.cluster.local",
#             user="root",
#             password=""
#         )
#         self.cursor = self.cnx.cursor()

#     def create_database(self):
#         try:
#             self.cursor.execute(
#                 "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
#         except mysql.connector.Error as err:
#             print("Failed creating database: {}".format(err))
#             exit(1)

#     def prepare 

        


#     def RegisterModelRequest(self, request, context):
#         cnt = 0
#         while True:
#             try:
#                 input = {}
#                 for key in request.input_batch:
#                     input[key] = torch.Tensor(np.load(BytesIO(request.input_batch[key]), allow_pickle=False)).int().to(device)
#                 # print(input)
#                 logits = self.engine(**input).logits
#                 # print(logits)
#                 logits = logits.cpu().detach().numpy()
#                 break
#             except:
#                 sleep(0.001 * 2 ** cnt)
#                 cnt += 1
#                 continue

#         output = BytesIO()
#         np.save(output, logits, allow_pickle=False)

#         return ModelInferenceResponse(
#             head=request.head,
#             retcode=0,
#             logits=output.getvalue(),
#             power=measure_gpu_power(),
#         )

#     async def run_tasks(self):
#         await asyncio.wait([
#             self.loop.create_task(self.monitor_gpu_power()), 
#             self.loop.create_task(self.delegate_inference()),
#         ])

#     def serve(self, address, n_threads):
#         server = grpc.server(ThreadPoolExecutor(max_workers=n_threads))
#         add_ModelInferenceServicer_to_server(self, server)
#         server.add_insecure_port(address)
#         server.start()
#         self.loop.run_until_complete(self.run_tasks())
#         server.wait_for_termination()

#     async def monitor_gpu_power(self):
#         while True:
#             self.gpu_powers.append(measure_gpu_power())
#             await asyncio.sleep(0.01)


zk = KazooClient(hosts='zk-0.zk-hs.utils.svc.cluster.local')
zk.start()

# Ensure a path, create if necessary
zk.ensure_path("/ecosys/model/")

# Create a node (deployment namespace)
zk.create("/ecosys/model/0", b"test")

