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

#[derive(Serialize)]
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

#[post("/nlpinfer")]
async fn index(
    model_input: web::Json<ModelInput>,
    // model: web::Data<CModule>,
    // tokenizer: web::Data<Tokenizer>,
    appdata: web::Data<AppState>,
) -> impl Responder {
    let model = appdata.model.clone();
    let tokenizer = appdata.tokenizer.clone();


    let user_sentence: &String = &model_input.sentence;
    let sentence_str = user_sentence.clone();

    let encoding = tokenizer.encode(sentence_str, false).unwrap();

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

    let result = model.forward_ts(&[input_ids, attention_mask]).unwrap();
    let result_vector = Vec::<f32>::from(result);

    let model_name = "test_bert".to_string();
    ModelOutput {
        sentence: user_sentence.clone(),
        model_name: model_name.clone(),
        vector: result_vector,
    }
}

#[derive(Clone)]
struct AppState {
    model: Arc<CModule>,
    tokenizer: Arc<Tokenizer>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let appdata = AppState {
        model: Arc::new(
            jit::CModule::load(
                "/Users/huzheng/PycharmProjects/Rust4SenVec/nlp_model/traced_bert.pt",
            )
            .unwrap(),
        ),
        tokenizer: Arc::new(Tokenizer::from_pretrained("bert-base-cased", None).unwrap()),
    };

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(appdata.clone()))
            .service(index)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
