
from ecosys.context.srv_ctx import ServiceContext
from protos.ecosys_pb2 import Head, Message, ResponseCode, RetCode
from ..utils.message import NumGenerator


class BaseHandler():

    req_msg = Message()
    rsp_msg = Message()

    def __init__(self, ctx: ServiceContext) -> None:
        self.ctx = ctx
        # self.make_request()
        # self.make_response()

    def check_req_type(self):
        req_type = self.req_msg.body.WhichOneof('payload')
        if req_type != self.req_name:
            err_msg = "wrong message type %s, should be %s" % (
                req_type, self.req_name)
            self.make_response_code(
                RetCode.ERR_MISMATCH_MESSAGE,
                err_msg,
            )
            self.ctx.logger.error(err_msg)
            return False
        return True

    def create_head(self):
        return Head(
            ctx_id=self.ctx.ctx_id,
            random_num=NumGenerator.randno(),
            flow_no=NumGenerator.flowno(),
            session_no=NumGenerator.session()
        )

    def req(self):
        # req_name = self.req_msg.body.WhichOneof('payload')
        return getattr(self.req_msg.body, self.req_name)

    def rsp(self):
        return getattr(self.rsp_msg.body, self.rsp_name)

    def make_request(self):
        self.req_msg.head.CopyFrom(self.create_head())
        req_field = getattr(self.req_msg.body, self.req_name)
        req_field.CopyFrom(type(req_field)())

    def make_response(self):
        self.rsp_msg.head.CopyFrom(self.req_msg.head)
        rsp_field = getattr(self.rsp_msg.body, self.rsp_name)
        rsp_field.CopyFrom(type(rsp_field)())
        self.make_response_code(RetCode.SUCCESS, "")

        # named_fields = self.req_msg.DESCRIPTOR.fields_by_name
        # numbered_fields = self.req_msg.DESCRIPTOR.fields_by_number

        # req_name = self.req_msg.body.WhichOneof('payload')
        # req_field_descriptor = named_fields[req_name]

        # req_number = req_field_descriptor.number
        # rsp_number = req_number + 1

        # if rsp_number not in numbered_fields:
        #     rsp_number = 1

        # rsp_field = getattr(self.rsp_msg.body, numbered_fields[rsp_number].name)
        # rsp_field.CopyFrom(type(rsp_field)())

    def make_response_code(self, ret, err):
        rc = ResponseCode(
            retcode=ret,
            error_message=err,
        )
        self.rsp().rc.CopyFrom(rc)

    def log_rc(self):
        if self.rsp().rc.retcode != RetCode.SUCCESS:
            self.ctx.logger.error('%s ret:%s, err:%s', type(
                self.req()), self.rsp().rc.retcode, self.rsp().rc.error_message)
        else:
            self.ctx.logger.info('%s ret:%s, err:%s', type(
                self.req()), self.rsp().rc.retcode, self.rsp().rc.error_message)


class InferenceHandler(BaseHandler):

    req_name = 'inference_request'
    rsp_name = 'inference_response'

    def __init__(self, ctx) -> None:
        super().__init__(ctx)


class RegisterModelHandler(BaseHandler):
    req_name = 'register_model_request'
    rsp_name = 'register_model_response'

    def __init__(self, ctx) -> None:
        super().__init__(ctx)


class ReportMetricsHandler(BaseHandler):
    req_name = 'report_metrics_request'
    rsp_name = 'simple_response'

    def __init__(self, ctx) -> None:
        super().__init__(ctx)


class ReportMetaHandler(BaseHandler):
    req_name = 'report_meta_request'
    rsp_name = 'simple_response'

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
