use rust_bert::pipelines::translation::TranslationModelBuilder;
use rust_bert::pipelines::translation::Language;
use rust_bert::pipelines::common::ModelType;
use tch::Device;

fn main() {
    translation();
}

fn translation() {
    let model = TranslationModelBuilder::new()
        .with_device(Device::cuda_if_available())
        .with_model_type(ModelType::Marian)
        .with_source_languages(vec![Language::Portuguese])
        .with_target_languages(vec![Language::English])
        .create_model()
        .unwrap();

    let text: &str = "Está um dia bonito";

    let output = model.translate(&[text], None, Language::English)
        .unwrap();

    for sentence in output {
        println!("{sentence}");
    }
}
