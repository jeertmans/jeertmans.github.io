# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.5"

gem "html-proofer", "~> 5.0", group: :test

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:windows]

group :jekyll_plugins do
  gem 'jekyll-scholar'
  gem 'jekyll-mentions'
  gem 'jekyll-gist'
  gem 'jekyll_github_sample'
  gem 'jekyll-target-blank'
  gem 'jekyll-redirect-from'
end
