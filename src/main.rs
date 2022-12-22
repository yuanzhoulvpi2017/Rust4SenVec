use actix_web::{
    body::BoxBody, http::header::ContentType, post, web, App, HttpRequest, HttpResponse,
    HttpServer, Responder,
};

use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;

use tch::Tensor;
use tch::{jit, CModule};

use tokenizers::tokenizer::Tokenizer;

#[derive(Deserialize)]
struct ModelInput {
    sentence: String,
}

#[derive(Serialize, Debug)]
struct ModelOutput {
    sentence: String,
    model_name: String,
    vector: Vec<f32>,
}

// Responder
impl Responder for ModelOutput {
    type Body = BoxBody;

    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        // Create response and set content type
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

#[derive(Clone)]
struct InferModel {
    model_name: String,
    model: Arc<CModule>,
    tokenizer: Arc<Tokenizer>,
}

impl InferModel {
    fn new(tokenizer_name: String, model_name: String) -> Self {
        let tokenizer = Arc::new(Tokenizer::from_pretrained(tokenizer_name.clone(), None).unwrap());
        let model = Arc::new(jit::CModule::load(model_name.clone()).unwrap());

        Self {
            model_name: tokenizer_name.clone(),
            model: model,
            tokenizer: tokenizer,
        }
    }

    fn infer(&self, user_sentence: &String) -> ModelOutput {
        let encoding = self.tokenizer.encode(user_sentence.clone(), false).unwrap();

        let encoding_ids_temp = encoding
            .get_ids()
            .iter()
            .map(|&i| i as i32)
            .collect::<Vec<i32>>();
        let attention_mask_temp = encoding
            .get_attention_mask()
            .iter()
            .map(|&i| i as i32)
            .collect::<Vec<i32>>();

        let attention_mask = Tensor::of_slice2(&[attention_mask_temp]);
        let input_ids = Tensor::of_slice2(&[encoding_ids_temp]);

        let result = self.model.forward_ts(&[input_ids, attention_mask]).unwrap();
        let result_vector = Vec::<f32>::from(result);

        ModelOutput {
            sentence: user_sentence.clone(),
            model_name: self.model_name.clone(),
            vector: result_vector,
        }
    }
}

// fn main() {
//     let appdata = InferModel::new(
//         "bert-base-cased".to_string(),
//         "/Users/huzheng/PycharmProjects/Rust4SenVec/nlp_model/traced_bert.pt".to_string(),
//     );

//     let res = appdata.infer(&"hello world".to_string());
    
//     println!("{:?}", res);

// }



#[post("/nlpinfer")]
async fn index(
    model_input: web::Json<ModelInput>,
    appdata: web::Data<InferModel>,
) -> impl Responder {
    let appdata = appdata.clone();
    let result = appdata.infer(&model_input.sentence);
    result

}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let appdata = InferModel::new(
        "bert-base-cased".to_string(),
        "/Users/huzheng/PycharmProjects/Rust4SenVec/nlp_model/traced_bert.pt".to_string(),
    );

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(appdata.clone()))
            .service(index)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
