using HerringstackApi.Abstractions;
using HerringstackApi.Services;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace HerringstackApi.Controllers;

[Route("api/[controller]")]
[ApiController]
public class SubjectsController : ControllerBase
{
    private readonly ICxr8DataManager _dataManager;

    public SubjectsController(ICxr8DataManager dataManager)
    {
        _dataManager = dataManager;
    }

    // GET: api/<LatentCxrImageController>/subjects
    [HttpGet("subjects")]
    public async Task<IActionResult> Get([FromQuery] int pageSize, [FromQuery] int pageNumber)
    {
        var subjectItems = await _dataManager.GetSubjectItemsAsync(pageSize, pageNumber);
        return Ok(subjectItems);
    }

    // GET api/<LatentCxrImageController>/subjects/5/images
    [HttpGet("subjects/{id}/images")]
    public async Task<IActionResult> GetSubjectImages(int id)
    {
        var images = await _dataManager.GetImagesForSubjectAsync(id);
        return Ok(images);
    }
}
